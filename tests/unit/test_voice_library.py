# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
import os
import wave
from dataclasses import asdict

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from multi_modal_ai_studio.core.voice_library import VoiceLibrary, validate_wav, MAX_AUDIO_BYTES
from multi_modal_ai_studio.config.schema import SessionConfig, TTSConfig
from multi_modal_ai_studio.backends.base import ConfigError
from multi_modal_ai_studio.backends.tts.openai_rest import build_speech_request, OpenAIRestTTSBackend
from multi_modal_ai_studio.webui.server import WebUIServer


def wav_bytes(rate=24000, seconds=1, channels=1, width=2):
    output = io.BytesIO()
    with wave.open(output, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(b'\0' * int(seconds * rate * channels * width))
    return output.getvalue()


def clone_config(**kwargs):
    return TTSConfig(**(dict(scheme='openai-rest', model='Qwen/Qwen3-TTS-12Hz-1.7B-Base',
                           api_base='http://localhost:18086/v1', request_profile='qwen3-base',
                           sample_rate=24000, language='en', instructions='') | kwargs))


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setenv('VAW_VOICE_LIBRARY_DIR', str(tmp_path / 'voices'))
    return VoiceLibrary()


def test_library_private_atomic_roundtrip_and_delete(library):
    audio = wav_bytes()
    entry = library.save(audio, name='Robot', transcript='Hello.', consent=True)
    assert len(entry['id']) == 32 and entry['duration'] == 1
    assert 'audio_base64' not in entry and 'consent' not in entry
    assert library.list() == [entry]
    assert library.load(entry['id'])[1] == audio
    assert os.stat(library.root / (entry['id'] + '.json')).st_mode & 0o777 == 0o600
    assert os.stat(library.root).st_mode & 0o777 == 0o700
    assert list(library.root.glob('.upload-*')) == []
    library.delete(entry['id'])
    assert library.list() == []
    with pytest.raises(ValueError, match='unavailable'):
        library.load(entry['id'])


@pytest.mark.parametrize('audio', [b'not WAV', wav_bytes(seconds=.5), wav_bytes(seconds=31),
    wav_bytes(channels=2), wav_bytes(width=1), wav_bytes(rate=4000), wav_bytes()[:-2], b'x' * (MAX_AUDIO_BYTES + 1)])
def test_bad_audio_rejected(audio):
    with pytest.raises(ValueError):
        validate_wav(audio)


@pytest.mark.parametrize('overrides', [dict(consent=False), dict(name=''), dict(name='x'*81),
                                    dict(transcript='x'*2001), dict(source='remote-url')])
def test_bad_metadata_rejected(library, overrides):
    with pytest.raises(ValueError):
        library.save(wav_bytes(), **(dict(name='Robot', consent=True) | overrides))


@pytest.mark.parametrize('voice_id', ['../secret', '/etc/passwd', 'https://example.com/a.wav', 'A'*32, ''])
def test_no_client_paths_or_urls(library, voice_id):
    with pytest.raises(ValueError):
        library.load(voice_id)


def test_symlink_and_corrupt_entries_not_listed(library, tmp_path):
    library.root.mkdir()
    target = tmp_path / 'outside.json'
    target.write_text('{}')
    (library.root / ('a'*32 + '.json')).symlink_to(target)
    (library.root / ('b'*32 + '.json')).write_text('broken')
    assert library.list() == []
    with pytest.raises(ValueError):
        library.delete('a'*32)
    assert target.exists()


@pytest.mark.parametrize('mode', ['icl', 'x-vector'])
def test_base_payload_and_config_do_not_embed_audio(library, tmp_path, mode):
    entry = library.save(wav_bytes(), name='Robot', transcript='Hello.', consent=True)
    config = clone_config(reference_voice_id=entry['id'], reference_voice_mode=mode, voice='Display name only')
    payload = build_speech_request(config, 'New words.')
    assert payload['task_type'] == 'Base' and payload['input'] == 'New words.'
    assert payload['stream'] and payload['stream_format'] == 'audio'
    assert payload['language'] == 'English'
    assert 'voice' not in payload and 'instructions' not in payload
    assert payload['x_vector_only_mode'] == (mode == 'x-vector')
    assert ('ref_text' in payload) == (mode == 'icl')
    assert base64.b64decode(payload['ref_audio'].split(',')[1]) == wav_bytes()
    session = SessionConfig.from_dict({'tts': asdict(config)})
    path = tmp_path / 'preset.yaml'
    session.to_yaml(path)
    restored = SessionConfig.from_yaml(path)
    assert restored.tts.reference_voice_id == entry['id']
    assert restored.tts.reference_voice_mode == mode
    assert 'base64' not in path.read_text()


def test_base_missing_transcript_requires_xvector_and_missing_reference_fails(library):
    entry = library.save(wav_bytes(), name='No transcript', consent=True)
    with pytest.raises(ConfigError, match='exact reference transcript'):
        OpenAIRestTTSBackend(clone_config(reference_voice_id=entry['id']))
    OpenAIRestTTSBackend(clone_config(reference_voice_id=entry['id'], reference_voice_mode='x-vector'))
    with pytest.raises(ConfigError):
        OpenAIRestTTSBackend(clone_config())
    with pytest.raises(ConfigError):
        OpenAIRestTTSBackend(clone_config(reference_voice_id=entry['id'], instructions='Make it happy'))


@pytest.mark.asyncio
async def test_reference_api_upload_list_audio_delete_and_limits(library):
    # Actual multipart reader and response serialization, not a mocked parser.
    handler = WebUIServer.__new__(WebUIServer)
    app = web.Application()
    app.router.add_get('/voices', handler.handle_reference_voices)
    app.router.add_post('/voices', handler.handle_reference_voice_upload)
    app.router.add_get('/voices/{voice_id}/audio', handler.handle_reference_voice_audio)
    app.router.add_delete('/voices/{voice_id}', handler.handle_reference_voice_delete)
    async with TestClient(TestServer(app)) as client:
        def upload(audio=None, consent='true'):
            form = aiohttp.FormData()
            form.add_field('audio', audio if audio is not None else wav_bytes(), filename='../../bad.wav')
            form.add_field('name', 'Test robot')
            form.add_field('consent', consent)
            return form
        response = await client.post('/voices', data=upload(), headers={'Origin': 'https://untrusted.example'})
        assert response.status == 400 and library.list() == []
        response = await client.post('/voices', data=upload(consent='false'))
        assert response.status == 400
        response = await client.post('/voices', data=upload())
        assert response.status == 201
        entry = await response.json()
        assert 'audio_base64' not in entry
        assert (await (await client.get('/voices')).json())['voices'] == [entry]
        response = await client.get('/voices/' + entry['id'] + '/audio')
        assert response.content_type == 'audio/wav' and await response.read() == wav_bytes()
        assert response.headers['Cache-Control'] == 'no-store'
        # >1MiB valid audio is supported, but the explicit 3MiB cap is enforced.
        response = await client.post('/voices', data=upload(wav_bytes(seconds=25)))
        assert response.status == 201
        response = await client.post('/voices', data=upload(b'x' * (MAX_AUDIO_BYTES + 1)))
        assert response.status == 400
        response = await client.delete('/voices/' + entry['id'], headers={'Origin': 'https://untrusted.example'})
        assert response.status == 400
        response = await client.delete('/voices/' + entry['id'])
        assert response.status == 200
        assert (await client.get('/voices/' + entry['id'] + '/audio')).status == 404

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faster_whisper_server.audio import Audio, AudioStream
from faster_whisper_server.text_utils import Transcription, common_prefix, to_full_sentences, word_to_text

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from faster_whisper_server.api_models import TranscriptionWord
    from faster_whisper_server.asr import FasterWhisperASR

logger = logging.getLogger(__name__)


class LocalAgreement:
    def __init__(self) -> None:
        self.unconfirmed = Transcription()

    def merge(self, confirmed: Transcription, incoming: Transcription) -> list[TranscriptionWord]:
        # https://github.com/ufal/whisper_streaming/blob/main/whisper_online.py#L264
        incoming = incoming.after(confirmed.end - 0.1)
        prefix = common_prefix(incoming.words, self.unconfirmed.words)
        logger.debug(f"Confirmed: {confirmed.text}")
        logger.debug(f"Unconfirmed: {self.unconfirmed.text}")
        logger.debug(f"Incoming: {incoming.text}")

        if len(incoming.words) > len(prefix):
            self.unconfirmed = Transcription(incoming.words[len(prefix) :])
        else:
            self.unconfirmed = Transcription()

        return prefix


# TODO: needs a better name
def needs_audio_after(confirmed: Transcription) -> float:
    full_sentences = to_full_sentences(confirmed.words)
    return full_sentences[-1][-1].end if len(full_sentences) > 0 else 0.0


def prompt(confirmed: Transcription) -> str | None:
    """Dynamic prompt based on confirmed transcription."""
    sentences = to_full_sentences(confirmed.words)
    return word_to_text(sentences[-1]) if len(sentences) > 0 else None


async def audio_transcriber(
    asr: FasterWhisperASR,
    audio_stream: AudioStream,
    min_duration: float,
) -> AsyncGenerator[Transcription, None]:
    """Real-time transcription - process each chunk independently like OpenAI"""
    chunk_count = 0
    current_sentence_words = []
    
    async for chunk_data in audio_stream.chunks(min_duration):
        chunk_count += 1
        
        # Create independent audio chunk - no cumulative processing
        chunk_audio = Audio(chunk_data, start=chunk_count * min_duration)
        
        # Transcribe chunk with no prompt to avoid hallucinations
        transcription, info = await asr.transcribe(chunk_audio, None)
        
        # Skip chunks with no actual speech (check if empty or just noise)
        text = transcription.text.strip()
        if not text or len(text) < 2:
            logger.debug(f"Chunk {chunk_count}: no meaningful text")
            continue
            
        if transcription.words:
            # Add words to current sentence
            current_sentence_words.extend(transcription.words)
            
            # Check if we have a sentence boundary (punctuation or silence)
            text = transcription.text.strip()
            if text and (text.endswith('.') or text.endswith('!') or text.endswith('?')):
                # Complete sentence - yield and reset
                complete_sentence = Transcription(current_sentence_words)
                logger.info(f"Complete sentence: '{complete_sentence.text}'")
                yield complete_sentence
                current_sentence_words = []
            else:
                # Partial sentence - yield current progress
                partial_sentence = Transcription(current_sentence_words)
                logger.debug(f"Partial: '{partial_sentence.text}'")
                yield partial_sentence
        else:
            logger.debug(f"Chunk {chunk_count}: no speech detected")
    
    # Yield any remaining words as final sentence
    if current_sentence_words:
        final_sentence = Transcription(current_sentence_words)
        logger.info(f"Final sentence: '{final_sentence.text}'")
        yield final_sentence
    
    logger.info("Audio transcriber finished")

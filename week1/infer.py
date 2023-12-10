import logging

from main import inference

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Enter a word to find its similar words")
    word = input()
    results = inference.infer(word)
    for word, score in results:
        print(f"{score:.3f}: {word}")

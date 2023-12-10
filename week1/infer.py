import logging

from train import infer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Enter a word to infer its vector")
    results = infer(input())
    for word, score in results:
        print(f"{score:.3f}: {word}")

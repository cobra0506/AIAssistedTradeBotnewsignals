from .multi_context_pipeline import train_multi_context
from .train_pipeline import TrainingConfig


def main():
    # This wrapper keeps a simple entrypoint for IDE/CLI usage.
    summary = train_multi_context(training_config=TrainingConfig())
    print(summary)


if __name__ == "__main__":
    main()

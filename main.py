from config import Config
from experiment import OptimizedExperiment
import asyncio


async def main():
    config = Config()

    experiment = OptimizedExperiment(config)

    # Run the experiment
    await experiment.run()

    print("Experiment completed! Results are stored in the results database.")


if __name__ == "__main__":
    asyncio.run(main())

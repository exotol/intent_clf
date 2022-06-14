import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(version_base="1.2", config_path=None, config_name="train.yaml")
def main(config: DictConfig):

    # # Imports can be nested inside @hydra.main to optimize tab completion
    # # https://github.com/facebookresearch/hydra/issues/934
    from rtk_mult_clf import utils
    from rtk_mult_clf.training_pipeline_sklearn import train

    #
    # # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()

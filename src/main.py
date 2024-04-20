import argparse
from Data_collecting import collect_data
from Data_preparation import prepare_data
from Data_validation import validate_data
from Model_creation import create_model
from Model_evaluation import model_evaluation
from Model_save import model_save

DEFAULT_FILE_PATH = "../data/obesity.csv"
SAVE_DIRECTORY_PATH = "../"


def main(args: argparse.Namespace) -> None:
    md = args.max_depth if args.max_depth is not None else 30  # 5, 2, 100
    msl = args.min_samples_leaf if args.min_samples_leaf is not None else 5
    mss = args.min_samples_split if args.min_samples_split is not None else 2
    ne = args.n_estimators if args.n_estimators is not None else 100

    obesity = collect_data.collect_data(args.file_path)
    prepare_data.data_preparation(obesity)
    validate_data.validate_column_types(obesity)
    validate_data.validate_categorical_column_ranges(obesity)
    model_bean = create_model.create(obesity, md, msl, mss, ne)
    model_evaluation.evaluate_model(model_bean, args.api, md, msl, mss, ne)
    model_save.save_model(model_bean.rfc, SAVE_DIRECTORY_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--file-path",
        required=True,
        type=str,
        default=DEFAULT_FILE_PATH,
        help="Input video file",
    )
    parser.add_argument(
        "-k",
        "--api",
        type=str,
        required=True,
        help="API key to login to Weights and Biases",
    )
    parser.add_argument(
        "-md", "--max_depth", type=int, required=False, help="Classifier max depth"
    )
    parser.add_argument(
        "-msl",
        "--min_samples_leaf",
        type=int,
        required=False,
        help="Classifier min samples leaf",
    )
    parser.add_argument(
        "-mss",
        "--min_samples_split",
        type=int,
        required=False,
        help="Classifier min samples split",
    )
    parser.add_argument(
        "-ne",
        "--n_estimators",
        type=int,
        required=False,
        help="Classifier n estimators",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())

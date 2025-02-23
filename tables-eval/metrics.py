from argparse import ArgumentParser
from teds_metric import teds_from_json
from df_metric import jaccord_from_json


def main(args):
    html_path = f"results/{args.model_name}_html.json"
    html_result = teds_from_json(html_path)

    csv_path = f"results/{args.model_name}_csv.json"
    csv_result = jaccord_from_json(csv_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemini")
    args = parser.parse_args()
    main(args)

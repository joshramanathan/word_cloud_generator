import os
import json
import pandas as pd
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download import CommonCrawlDownloadExtractStage
from nemo_curator.stages.text.io.writer import JsonlWriter
from . utilities import clean_text


class WordCloudGenerator:
    def __init__(self, data_dir: str = None, language: str = "ENGLISH") -> None:

        if not data_dir:
            self.data_dir = os.path.join('.', "data")
        else:
            self.data_dir = data_dir

        self.language = language


    def get_cc_data(self, snapshot: str = "2026-12") -> None:
        """
        Get Common Crawl data and save it to a JSONL file.
        """

        # Create pipeline
        pipeline = Pipeline(
            name="cc_data_pipeline",
            description="Pipeline to download and extract Common Crawl data"
        )
        output_path = os.path.join(self.data_dir, "cc_jsonls")
        warc_path = os.path.join(self.data_dir, "cc_warcs")

        # Add Common Crawl stage
        cc_stage = CommonCrawlDownloadExtractStage(
            start_snapshot=snapshot,
            end_snapshot=snapshot,
            download_dir=warc_path,
            crawl_type="main",
            use_aws_to_download=False,
            url_limit=100,
            record_limit=1000
        )
        pipeline.add_stage(cc_stage)

        # Add JSONL writer stage
        writer = JsonlWriter(output_path)
        pipeline.add_stage(writer)

        # Run pipeline
        results = pipeline.run()
        print(f"  Saved {len(results)} records to {output_path}")


    def preprocess_data(self) -> None:
        """
        Preprocess the Common Crawl data by extracting English, deduplicating, and
        validating the content. Saves the cleaned data to a DataFrame feather file.
        """

        # Aggregate JSONL and convert to DataFrame
        cc_jsonl_dir = os.path.join(self.data_dir, "cc_jsonls")
        data = []
        for file in os.listdir(cc_jsonl_dir):
            if file.endswith(".jsonl"):
                file_path = os.path.join(cc_jsonl_dir, file)
                with open(file_path, "r") as f:
                    for line in f:
                        record = json.loads(line)
                        if record.get("language") == self.language and record.get("text"):
                            data.append({
                                "url": record.get("url"),
                                "text": clean_text(record.get("text"))
                            })
        df = pd.DataFrame(data)
        # df.dropna(subset=["text"], inplace=True)

        total_records = len(df)
        # print(f"Total records before cleaning: {total_records}")

        # Drop duplicates and empties
        df.drop_duplicates(subset=["text"], inplace=True)

        # Save the cleaned DataFrame to a feather file
        cc_df_path = os.path.join(self.data_dir, "cc_data.feather")
        print(f"  Saving {len(df)} records to {cc_df_path}")
        df.to_feather(cc_df_path)

        # Get stats
        word_count = df["text"].apply(lambda x: len(x.split())).sum()
        print(f"    Total word count: {word_count}")

        doc_count = len(df)
        print(f"    Total document count: {doc_count}")

        vocab_size = len(set(" ".join(df["text"].tolist()).split()))
        print(f"    Vocabulary size: {vocab_size}")

        duplicate_rate = (total_records - len(df)) / total_records
        print(f"    Duplicate rate: {duplicate_rate}")

        ttr = vocab_size / word_count
        print(f"    TTR: {ttr}")
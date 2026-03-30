from word_cloud_generator.generator import WordCloudGenerator

if __name__ == "__main__":
    generator = WordCloudGenerator()

    print("Getting Common Crawl data...")
    generator.get_cc_data()

    print("Preprocessing data...")
    generator.preprocess_data()
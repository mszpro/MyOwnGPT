import deepl
import os

client = deepl.Translator('9ce030ce-2b6a-e616-143c-d1c5e330be4b')

dataSetFileNames = os.listdir("/Users/msz/Downloads/MyOwnGPT/dataset")
dataSetFileNames = ["XcodeInAppPurchaseTesting.md"]

for filename in dataSetFileNames:
    file_path = "/Users/msz/Downloads/MyOwnGPT/dataset/" + filename
    print("Translating file", filename)
    with open(file_path, 'r') as file:
        markdown_text = file.read()
        result = client.translate_text(markdown_text, target_lang='JA')
        print(result.text)
        with open("/Users/msz/Downloads/MyOwnGPT/dataset_ja/" + filename, "w") as file:
            file.write(result.text)
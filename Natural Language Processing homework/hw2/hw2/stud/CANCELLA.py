from data_utils import AbsaDataModule

positive = 0
negative = 0
neutral = 0
conflict = 0

module = AbsaDataModule()
module.setup()
for el in module.validation_dataset:
    for target in el.targets:
        sent = target.sentiment
        if sent == "positive":
            positive += 1
        elif sent == "negative":
            negative += 1
        elif sent == "neutral":
            neutral += 1
        elif sent == "conflict":
            conflict += 1
        else:
            print(sent)

print(f"Positive: {positive}")
print(f"negative: {negative}")
print(f"neutral: {neutral}")
print(f"conflict: {conflict}")




class Amz_data:
    def parse(self, path):
        import json
        with open(path, "r") as ins:
            for line in ins:
                obj = json.loads(line)
                reviewText = obj["reviewText"]
                summary = obj["summary"]
                reviewID = obj["reviewID"]
                overall = obj["overall"]
                asin = obj["asin"]
                print(obj)



ad = Amz_data()
ad.parse("J:\\reviews_Books.json")

import json
import os

from util.text_process import TextProcess

home = os.environ["HOME"]
path = "".join((home, "/data/aan/papers_text/"))
path_processed = "".join((home, "/data/aan/paper_processed_nostem.txt"))
f_processed = open(path_processed, "w")

business_id = "aRkYtXfmEKYG-eTDf_qUsw"
batch = ""

for root, dirs, filenames in os.walk(path):
    for filename in filenames:
        if filename[-3:] == "txt":
            temp_path = "".join((root, filename))
            temp_f = open(temp_path, "r")
            for temp_line in temp_f:
                text = TextProcess.process(temp_line)
                text = "".join((text, "\n"))
                line_processed = "\t".join((business_id, text))
                batch = "".join((batch, line_processed))
                if len(batch) > 100000:
                    f_processed.write(batch)
                    f_processed.write("\n")
                    batch = ""
f_processed.write(batch)
f_processed.write("\n")




# f = open(path, "r")
# f_processed = open(path_processed, "w")
# batch = ""
# for line in f:
#     obj = json.loads(line)
#     text = TextProcess.process(obj["text"])
#     user_id = str(obj["user_id"])
#     business_id = str(obj["business_id"])
#     stars = str(obj["stars"])
#
#     # line_processed = "\t".join((user_id, business_id, stars, text))
#     line_processed = "\t".join((business_id, text))
#     batch = "".join((batch, line_processed))
#     if len(batch) > 100000:
#         f_processed.write(batch)
#         f_processed.write("\n")
#         batch = ""
# f_processed.write(batch)
# f_processed.write("\n")
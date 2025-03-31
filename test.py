import os
import xml.etree.ElementTree as ET

root_ = r"D:\anshul\remoteHR\mahnob\Sessions"
subjects_dir = r"D:\anshul\remoteHR\mahnob\Subjects"

# subject_info = {}
# for s in os.listdir(subjects_dir):
#     subject_xml = os.path.join(subjects_dir, s)
#
#     # Load and parse the XML file
#     tree = ET.parse(subject_xml)  # Replace 'your_file.xml' with the actual XML file path
#     root = tree.getroot()
#
#     # # Print the root element and the structure
#     # print("Root element:", root.tag)
#     # print("Root attributes:", root.attrib)
#
#     subject_info[root.attrib["id"]] = {
#         "s": root.attrib["gender"],
#         "e": root.attrib["ethnicity"],
#         "d": root.attrib["dob"]
#     }
#
# print(subject_info)
#
# sessions = []
# subjects = []
# age = []
# sex = []
# ethnicity = []
#
# from datetime import datetime
#
# def get_age(dob, cd):
#
#     # Define the two date strings
#     dob_str = dob
#     current_date_str = cd
#
#     # Convert strings to datetime objects
#     dob = datetime.strptime(dob_str, "%Y-%m-%d")  # For "1983-7-3"
#     current_date = datetime.strptime(current_date_str, "%Y-%m-%d")  # For "2009-07-23"
#
#     # Calculate the age
#     age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
#     return age
#
#
# for s in os.listdir(root_):
#     session_dir = os.path.join(root_, s)
#     xml_info = os.path.join(session_dir, "session.xml")
#     tree = ET.parse(xml_info)  # Replace 'your_file.xml' with the actual XML file path
#     root = tree.getroot()
#
#     # Print the root element and the structure
#     session_date = root.attrib["date"]
#     for child in root:
#         if child.tag == "subject":
#             subject = child.attrib['id']
#
#     sessions.append(s)
#     subjects.append(subject)
#     sex.append(subject_info[subject]["s"])
#     ethnicity.append(subject_info[subject]["e"])
#     age.append(get_age(subject_info[subject]["d"], session_date))

import pandas as pd
# df = pd.DataFrame()
# df["session"] = sessions
# df["subject"] = subjects
# df["sex"] = sex
# df["ethnicity"] = ethnicity
# df["age"] = age
# print(df.head())
# df.to_csv("mahnob_session_info.csv", index=False)

df = pd.read_csv("mahnob_session_info.csv")
males_fold1 = []
males_fold2 = []
females = []
for _, row in df.iterrows():
    if row["sex"] == "Male":
        if row["subject"] % 2 == 0:
            males_fold1.append(str(row["session"]))
        else:
            males_fold2.append(str(row["session"]))
    else:
        females.append(str(row["session"]))

print(f"MALE FOLD 1: {males_fold1}")
print("\n\n")
print(f"MALE FOLD 2: {males_fold2}")
print("\n\n")
print(f"FEMALE: {females}")

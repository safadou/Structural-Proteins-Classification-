"""

Config file for Streamlit App

"""

from member import Member

root_dir='../'

img_dir = root_dir+'images/'
models_dir = root_dir+'models/'
data_dir = root_dir+'data/'

TITLE = "Structural Protein Classification"

TEAM_MEMBERS = [
    Member(
        name="DIALLO Sadou Safa",
        linkedin_url="https://www.linkedin.com/in/sadou-safa-diallo-a0839b49/",
        github_url="https://github.com/safadou",

    ),
    Member(
        name="NGIZULU Edi",
        linkedin_url="https://www.linkedin.com/in/edi-ngizulu-57256316a/",
        github_url="https://github.com/nedikas",
    ),
]

PROMOTION = "Formation Continue Data Scientist - May 2021"

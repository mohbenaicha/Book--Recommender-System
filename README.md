# Book-Recommender

This is a Python package and API I built for recommending books to users based on millions of user reviews that have I have scraped. The package and API are kept in this single repo to facilitate access to them; otherwise, the package is downloadable through `pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ book-recommender==0.0.1` for the latest version and the API can be launched locally as follows:

1. clone this repo `git init`, `git clone https://github.com/mohbenaicha/Book--Recommender-System`
2. create new environment such as: `conda create --name newenv`
3. setup tox `pip install tox`
4. cd into the appropriate directory then cd into recommender-api/: `cd Book--Recommender-System/recommender-api/`
5.  `tox -e run` (give it some time to setup dependecies and including book-recommender)
6.  use the link provided by uvicorn to access the api's GUI

#### Using the API's graphical UI:
- Open the 'proceed' link in a new tab and keep the existing tab to follow the rest of the instructions.
- Expand the 'POST' request section
- Find the 'try it out' button.
- Edit the 'Request body' input box based on the user_id of your liking, then hit 'execute'.
- Scroll down further (ignore the cURL request template following it)
- Check out the recommendations under the reponse body.


#### Using the API through cURL:
1. While the app is hosted, run the following command from your Bash shell or shell that supports cURL:

```
curl -X 'POST' \
  'http://localhost:8001/api/v1/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": [
    {
      "user_id": "1"
    }
  ]
}'
```

#### Using the Django-based api

1. clone this repo `git init`, `git clone https://github.com/mohbenaicha/Book--Recommender-System`
2. cd into the appropriate directory then cd into django-based-api/: `cd Book--Recommender-System/django-based-api/`
3. create new environment such as: `conda create --name newenv`
4. setup API dependencies `pip install -rrequirements.txt`
5. launch Python; the recommender model features a large neural net which needs to be unzipped to work properly:
 > In Python (with environment activated):
 ``` 
 from book_recommender.utilities.data_manager import zip_unzip_model
 
 zip_unzip_model(test=False, zip=False) # unzips model
 ```
6. Back in your shell, ensure you are in ./django-based-api, use the following command to launch and test the API in a local development server:
`python manage.py runserver 0.0.0.0:8001` (or use another free port)
7. Access the API through: http://127.0.0.1:8001/api/recommend/
8. Under the raw data form, a list of string user ids can be inserted to get the recommendations:
```
{
    "inputs": ["52", "100023", "389"]
}
```

## Alternatively, get recommendations manually:
1. create new environment such as: `conda create --name newenv`
2. `pip install --extra-index-url https://pypi.fury.io/mohbenaicha/ book-recommender==0.0.1`
- launch python in the environment used above (`newenv`)
- import the package, unzip the model and make recommendations:

```
from recomender_model.utilities.data_manager import zip_unzip_model
from recomender_model.recommend import make_recommendation

zip_unzip_model(zip=False, test=False)

users = ['32', '100023'] # edit input data per your liking as type: list(str)
print(make_recommendation(input_data=users, test=False))
```

## Appendix: Notes and Disclaimers:

- Note 1: If you're using the API, status code 200 means the post request was valid and should yield recommendations within the response body for the user ids you've provided.
- Note 2: The recommender is trained using tensorflow_recommenders two tower model which is essentially collaborative filtering in its simplest form. The data used consists of over 1.7 million reviews on ~ 60K books (with over 700K unique users) that was collected over an extensive period of time. Given the model's large size, it is zipped and requires unzipping before it is used if recommendations are made through python script, otherwise the the API unzips the mode as soon as it is launched.
- Note 3: User IDs are annonymized (encoded) for obvious reasons. The research work folder contains details on how I make sense of the recommendations user genre overlap

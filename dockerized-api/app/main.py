from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from app.api import api_router
from app.config import settings, setup_app_logging
from recommender_model.utilities.data_manager import zip_unzip_model

# setup logging as early as possible
setup_app_logging(config=settings)


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

root_router = APIRouter()


@root_router.get("/")
def index(request: Request) -> Any:
    """Home end point"""
    body = (
        "<html>"
        "<body style='padding: 10px;'>"
        "<h1>Welcome to the Book Recommender API</h1>"
        """<div> You've reached the Book Recommender API:
        <a href='/docs'>Proceed</a> </div>"""
        "<div>"
        "<div>"
        "</div>"
        "Using the api interface:"
        "<ul>"
        """<li> Open the 'proceed' link in a new tab and keep this one to follow the
        rest of the instructions. </li>"""
        "<li> Expand the 'POST' request section</li>"
        "<li> Find the 'try it out' button. </li>"
        """<li> Edit the 'Request body' input box to the user of your liking then
        hit 'execute'. </li>"""
        "<li> Scroll down further (ignore the cURL request template following it) </li>"
        "<li> Check the recommended books under the reponse body .</li>"
        """<li> Note: status code 200 means the post request was valid and should
        yield recommendations within the response body. </li>"""
        "</ul>"
        "</div>"
        "</body>"
        "</html>"
    )

    return HTMLResponse(content=body)  # return a webpage response


# route to health and predict end points
app.include_router(api_router, prefix=settings.API_V1_STR)
# route to home end point
app.include_router(root_router)

# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        # admits various origins (protocols/ports/etc.) if front end and
        # backend attempt to communicate
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        # allows any request method (post/get)
        allow_methods=["*"],
        # allows request with any header like cookies/User-Agent; can disable
        # not browser user-agents for example
        allow_headers=["*"],
    )


def unzip_and_app(app_object: FastAPI = app):
    zip_unzip_model(test=False, zip=False)
    return app_object

app = unzip_and_app(app_object=app)


if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning("Running in development mode.")
    import uvicorn

    # log level does not follow production api log level since .run is
    # for debugging in development; production api captures INFO level
    # uvicorn logs using Loguru
    uvicorn.run(app, host="localhost", port=8001, log_level="debug")

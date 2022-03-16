# Swatch! API
This is the API that the [Swatch!](https://github.com/Kipwisp/swatch-app) web application utilizes to analyze images. It runs on a Flask server and utilizes Gunicorn as a WSGI. 

## Setting Up
To install the needed dependencies for the project, run the following command:
```
pip install -r requirements.txt
```

## Running
To start the server, run the following command:
```
flask run
```

## Request Format
The API takes in a single image for the body of a request.

## Response Format
``` JavaScript
{
    color_proportion: {
        {
            polar: number[],
            count: number,
            hsv: number[],
            hex: string,
            rgb: number[],
            pos: number[],
        }[]
    },
    color_palette: {
        {
            hex: string,
            pos: number[],
        }[]
    },
    value_distribution: {
        {
            count: number,
            bin: number,
        }[]
    },
}
```

## Licensing
This project is licensed under the GNU GPLv3 - see [LICENSE](https://raw.githubusercontent.com/Kipwisp/swatch-api/main/LICENSE?token=GHSAT0AAAAAABSOHVBT3CD4QXCCONOFH3GSYRRM3JQ) for details.

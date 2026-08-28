# Machine Learning FastAPI Service

FastAPI service exposing Machine Learning endpoints (regression and classification) with PostgreSQL persistence via SQLAlchemy/Alembic.

## Local setup (Linux/WSL)

```bash
# Create the python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the requirements
pip install -r requirements.txt
```

## Run the server

```bash
uvicorn machine_learning.main:app --reload --port 8080
```

Open the project in the browser:

- http://localhost:8080
- http://localhost:8080/docs

## Docker

```bash
# Build the image
docker build -t ml_fastapi .

# Run the container
docker run --rm -it -p 8080:8080 ml_fastapi
```

Useful commands inside the container:

```bash
ls
ls -a
```

## Database (PostgreSQL)

### Start the local PostgreSQL server (Ubuntu/WSL)

```bash
sudo service postgresql start
```

Local PostgreSQL database name for this project: **ai_recruitment**

### Query existing PostgreSQL databases in WSL

```bash
sudo -u postgres psql     # Enter the postgres environment
\l                         # List postgresql databases
\c database_name           # Enter a postgresql database
\dt                        # List the existing tables
\q                         # Exit
```

## Migrations (Alembic)

```bash
# Create the Alembic environment (run just once)
alembic init alembic

# Create a manual migration
alembic revision -m "Create chats table"

# Autogenerate a migration from the models
alembic revision --autogenerate -m "Create User table"

# Apply migrations
alembic upgrade head
```

## Resources and documentation

ML course:
- https://www.youtube.com/watch?v=xyU2pzKTQE0&t=14413s
- https://www.youtube.com/watch?v=Rgag-Clu5L4
- https://www.youtube.com/watch?v=TkN2i-_4N4g
- https://www.youtube.com/watch?v=CMEWVn1uZpQ

Additional documentation:
- https://www.w3schools.com/python/python_ml_getting_started.asp
- https://pll.harvard.edu/course/machine-learning-and-ai-python
- https://www.youtube.com/watch?v=1fcfZ_Ne8ok

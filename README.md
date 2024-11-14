# Create the pyton virtual enviroment
python3 -m venv venv

# Activate the virtual enviroment venv
source venv/bin/activate

# Start the local postgresql database in ubuntu (wsl)
sudo service postgresql start

# Install the requitement.txt
pip install -r requirements.txt

# Run the uvicorn server
uvicornmachine_learning.main:app --reload

# Open the project in the browser
localhost:8000
localhost:8000/docs


# Create alembic enviroment (run just one)
alembic init alembic

# create migrations
alembic revision -m "Create chats table"

# autogeneration migration
alembic revision --autogenerate -m "Create User table"

# run migration
alembic upgrade head



# Extra project documentation

# local postgresql database name from the ai_job project
ai_recruitment

# Consult existing postgresql databases in WSL
sudo -u postgres psql     # Enter in the postgres enviroment
\l                        # List postgresql database
\c database_name          # Enter in a postgresql database
\q                        # Exit from a postgresql database
\dt                       # List the existing tables



# Extra documentation

# Use the official Python base image with the desired version
FROM rocker/tidyverse:latest

# Register github token from secrets path
RUN --mount=type=secret,id=github_token \
  export GITHUB_PAT=$(cat /run/secrets/github_token)

# Set the working directory inside the container
WORKDIR /app

# Copy the local files into the container
COPY . /app

# # Clone the GitHub repository
# RUN git clone https://github.com/shahcompbio/scdna_replication_tools.git

# # Change to the cloned repository's directory
# WORKDIR /app/scdna_replication_tools

# # Create a virtual environment
# RUN python -m venv venv/

# # Activate the virtual environment
# RUN /bin/bash -c "source venv/bin/activate"


# Install the package in development mode
RUN R -e "install.packages('Seurat',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "install.packages('argparse',dependencies=TRUE, repos='http://cran.rstudio.com/')"

RUN apt-get update && apt-get install -y python3 python3-pip

RUN cp /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip

RUN pip install pyranges umap-learn ete3 hdbscan
RUN pip install .

# Expose any required ports
# EXPOSE <port_number>

# Run any required commands or scripts to start the application
# CMD <command>
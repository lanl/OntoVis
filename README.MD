# Autonomous Visualization: Ontologies(OntoVis)
The aim of this research project is to help LLMs in making correct scientific visualizations.

## Setup

The python environment:
```bash
uv python install 3.11
uv venv --python 3.11
uv venv .venv
source .venv/bin/activate
uv sync
uv pip install -e .  #maybe needed
uv run python -m ipykernel install --user --name venv_OntoVis --display-name "venv_OntoVis" # register the environment with Jupyter notebook 
```

neo4j setup:
```bash
mkdir workspace
cd workspace

docker pull neo4j:5  # neo4j in a container

# Run it:
docker run \
  --name neo4j-local \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4jpassword \
  -v $PWD/neo4j-data:/data \
  neo4j:5
```

Check by typing ```http://localhost:7474``` in a browser  
 - Username: neo4j  
 - Password: neo4jpassword  
or whatever was specified above


To stop the container:
```bash
docker stop neo4j-local
```

```docker rm neo4j-local``` will delete the container

and run it later:
```bash
docker start neo4j-local
```


docker stop:
docker-compose down --remove-orphans

Build docker:
docker-compose build --no-cache code-execution-api

docker-run:
docker-compose up code-execution-api


## Copyright
© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.


Status:
- docker + fastapi server should be fine
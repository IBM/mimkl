FROM python:3.6 
# system dependencies 
RUN apt-get -y update && apt-get install -y build-essential g++ gcc make git zip unzip \ 
      libopenblas-dev cmake && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
# mimkl installation
WORKDIR /workspace/mimkl
COPY . .
RUN if [ -d "build_" ]; then rm -r build_; fi
RUN MIMKL_CXX_FLAGS="-O3 -fopenmp" pip install -e .[testing]
WORKDIR /workspace
# get the entry point and make it executable
COPY docker/docker-entrypoint.sh /usr/bin/
RUN chmod +x /usr/bin/docker-entrypoint.sh
# define the entrypoint
ENTRYPOINT [ "/usr/bin/docker-entrypoint.sh" ]

 #!/bin/sh

echo ${DOCKER_PASSWORD} | docker login -u ${DOCKER_USERNAME} --password-stdin
docker tag tsenit/mimkl:latest tsenit/mimkl:${TRAVIS_COMMIT}
docker push tsenit/mimkl:${TRAVIS_COMMIT}
docker push tsenit/mimkl:latest

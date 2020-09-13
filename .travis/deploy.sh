 #!/bin/sh

echo ${DOCKER_PASSWORD} | docker login -u ${DOCKER_USERNAME} --password-stdin
docker tag ${DOCKER_USERNAME}/mimkl:latest ${DOCKER_USERNAME}/mimkl:${TRAVIS_COMMIT}
docker push ${DOCKER_USERNAME}/mimkl:${TRAVIS_COMMIT}
docker push ${DOCKER_USERNAME}/mimkl:latest

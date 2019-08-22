 #!/bin/sh

echo ${DOCKER_PASSWORD} | docker login -u ${DOCKER_USERNAME} --password-stdin
docker tag drugilsberg/mimkl:latest drugilsberg/mimkl:${TRAVIS_COMMIT}
docker push drugilsberg/mimkl:${TRAVIS_COMMIT}
docker push drugilsberg/mimkl:latest

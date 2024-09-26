FROM node:18

WORKDIR /app

# install IPFS Kubo
RUN wget https://dist.ipfs.tech/kubo/v0.23.0/kubo_v0.23.0_linux-amd64.tar.gz && \
    tar -xvzf kubo_v0.23.0_linux-amd64.tar.gz && \
    cd kubo && \
    bash install.sh

# install IPFS Cluster
RUN wget https://dist.ipfs.tech/ipfs-cluster-service/v1.0.7/ipfs-cluster-service_v1.0.7_linux-amd64.tar.gz && \
    wget https://dist.ipfs.tech/ipfs-cluster-ctl/v1.0.7/ipfs-cluster-ctl_v1.0.7_linux-amd64.tar.gz && \
    tar -xvzf ipfs-cluster-service_v1.0.7_linux-amd64.tar.gz && \
    tar -xvzf ipfs-cluster-ctl_v1.0.7_linux-amd64.tar.gz && \
    mv ipfs-cluster-service/ipfs-cluster-service /usr/local/bin && \
    mv ipfs-cluster-ctl/ipfs-cluster-ctl /usr/local/bin

# install PM2
RUN npm install -g pm2

# install deps
COPY ts/package.json /app/ts/package.json
COPY ts/pnpm-lock.yaml /app/ts/pnpm-lock.yaml

# pnpm install in ts dir
WORKDIR /app/ts
RUN pnpm install

WORKDIR /app

# copy in src
COPY ts/src /app/ts/src

COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
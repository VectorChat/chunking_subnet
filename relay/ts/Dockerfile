FROM node:20

WORKDIR /app

# install pnpm
RUN npm install -g pnpm

# install deps
COPY package*.json pnpm-lock.yaml* ./
RUN pnpm install

COPY src ./src
RUN mkdir -p ./src/__generated__/interfaces
RUN pnpm gen-polkadot:finney


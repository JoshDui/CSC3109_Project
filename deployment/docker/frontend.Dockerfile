# syntax=docker/dockerfile:1

FROM oven/bun:1.3.14-alpine AS build

WORKDIR /app

COPY frontend/package.json frontend/bun.lock ./
RUN bun install --frozen-lockfile

COPY frontend/index.html ./index.html
COPY frontend/tsconfig.json ./tsconfig.json
COPY frontend/vite.config.ts ./vite.config.ts
COPY frontend/public ./public
COPY frontend/scripts ./scripts
COPY frontend/src ./src

RUN bun run build:cdn

FROM caddy:2.10.2-alpine AS runtime

COPY docker/Caddyfile /etc/caddy/Caddyfile
COPY --from=build /app/dist /srv

EXPOSE 8080

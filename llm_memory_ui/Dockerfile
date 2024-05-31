FROM node:18-alpine as builder

ENV NODE_ROOT /usr/src/app/
RUN mkdir -p $NODE_ROOT
WORKDIR $NODE_ROOT

COPY ./package.json .
RUN npm install --production

COPY . .
RUN npm run build

FROM nginx:1.25.5-alpine

COPY --from=builder /usr/src/app/build /usr/share/nginx/html/memgpt

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

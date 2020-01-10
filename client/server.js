const serve = require('koa-static');
const Koa = require('koa');

const app = new Koa();

app.use(serve('static'));

app.listen(3000, () => console.log('Server is up'));
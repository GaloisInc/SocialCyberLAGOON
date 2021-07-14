import { createApp } from 'vue'
import vuetify from './plugins/vuetify'
import App from './App.vue'
import router from './router'

// eslint-disable-next-line
declare var VueSpaBackend: any;
createApp(App)
  .use(router)
  .use(vuetify)
  .use(VueSpaBackend)
  .mount('#app')

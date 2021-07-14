<template>
  <v-app>
    <v-main>
      <router-view/>
    </v-main>
  </v-app>
</template>

<style lang="scss">
  select {
    border: solid 1px #000;
    border-radius: 0.25em;
    cursor: pointer;
    padding: 0.1em;
  }
</style>

<script lang="ts">
import { defineComponent } from 'vue'

export default defineComponent({
  name: 'App',

  data () {
    return {
      //
    }
  },
})
</script>

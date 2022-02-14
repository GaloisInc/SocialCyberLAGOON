<template lang="pug">
div.vuespa-searchbar
  teleport(to="body")
    div.modal-container(v-if="open")
      .shade(@click="open=false")
      .modal(@click.stopPropagation)
        div(style="display: flex; flex-direction: row; margin-bottom: 1em")
          span(style="white-space: nowrap") ID|Regex ^
          input(ref="openInput" v-model="searchQuery" placeholder="Entity search"
              style="flex-grow: 1:"
              @focus="$event.target.select()")
        ul
          li(v-for="r of results" @click="open=false; $emit('update:modelValue', r.value)")
            span {{r.label}}

  div
    v-btn(outlined x-small @click="doOpen()") {{label}}
</template>
<style scoped lang="scss">
  .modal-container {
    position: fixed;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
  }

  .shade {
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    bottom: 0;
    background-color: #000;
    opacity: 0.3;
  }

  .modal {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    min-width: 30em;
    max-width: 80%;
    max-height: 90%;
    overflow-y: auto;
    background-color: white;
    border: solid 1px #444;
    border-radius: 0.5em;

    li {
      cursor: pointer;

      &:hover {
        background-color: #bbb;
      }
    }
  }

  input {
    border: solid 1px #ddd;
    padding: 0 0.25em;
    width: 100%;
    z-index: 11;
  }
</style>
<script lang="ts">
import {defineComponent} from 'vue';

export default defineComponent({
  props: {
    modelValue: {type: Number, required: true},
    endpointData: {type: String, required: true},
    endpointLabel: {type: String, required: true},
  },

  data() {
    return {
      label: '',  // cached version of resolved entity label
      open: false,
      results: [],
      resultsLoading: false,
      searchCallbackId: null as any,
      searchLatency: 300,
      searchQuery: '',
    };
  },

  mounted() {
    if (this.modelValue) {
      this.updateLabel();
    }
  },

  watch: {
    label() {
      if (!this.open) this.searchQuery = this.label;
    },
    modelValue() {
      this.updateLabel();
    },
    searchQuery() {
      if (!this.open) return;
      this.searchCallbackId && clearTimeout(this.searchCallbackId);
      this.searchCallbackId = setTimeout(() => this.search(),
          this.searchLatency);
    },
  },

  methods: {
    doOpen() {
      if (this.open) return;

      this.open = true;
      setTimeout(() => {
        (this.$refs.openInput as any).focus();
      }, 1);
    },
    search() {
      (async () => {
        this.resultsLoading = true;
        try {
          this.results = await this.$vuespa.call(this.endpointData,
              '^' + this.searchQuery);
          console.log(`Search returned ${this.results.length} results`);
        }
        finally {
          this.resultsLoading = false;
        }
      })().catch(console.error);
    },
    updateLabel() {
      (async () => {
        this.label = await this.$vuespa.call(this.endpointLabel, this.modelValue);
      })().catch(console.error);
    },
  },
})
</script>


<template>
  <temporal-view
      v-model:focus-id="focusIdModel"
      v-model:time-mean="timeMeanModel"
      v-model:time-range="timeRangeModel"
      v-model:graph-hops="graphHopsModel"
      />
</template>

<script lang="ts">
import { defineComponent } from 'vue';
import TemporalView from '../components/TemporalView.vue';
import router from '../router';

export default defineComponent({
  name: 'Home',

  components: {
    TemporalView,
  },

  props: {
    focusId: {type: String, required: true},
    timeMean: {type: String, required: true},
    timeRange: {type: String, required: true},
    graphHops: {type: String, required: true},
  },

  data() {
    return {
      navChange: {} as any,
      navUpdateId: null as any,
    };
  },

  computed: {
    focusIdModel: {
      get(): number { return +this.focusId; },
      set(v: number) { this.nav({focusId: v}); },
    },
    timeMeanModel: {
      get(): number { return +this.timeMean; },
      set(v: number) { this.nav({timeMean: v}); },
    },
    timeRangeModel: {
      get(): number { return +this.timeRange; },
      set(v: number) { this.nav({timeRange: v}); },
    },
    graphHopsModel: {
      get(): number { return +this.graphHops; },
      set(v: number) { this.nav({graphHops: v}); },
    },
  },

  methods: {
    nav(o: any) {
      // Collect all such emissions
      Object.assign(this.navChange, o);
      this.navUpdateId && clearTimeout(this.navUpdateId);
      this.navUpdateId = setTimeout(() => {
        router.push({path: '/', query: Object.assign({}, this.$route.query, this.navChange)});
        this.navChange = {};
      }, 10);
    },
  },
});
</script>

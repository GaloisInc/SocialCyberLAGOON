<template lang="pug">
div.temporalview
  div.cytoscape(ref="cytoscape")

  //- Width not working? See CSS
  v-dialog(v-model="focusDetails" persistent)
    v-card(style="display: flex; flex-direction: column; max-height: inherit")
      v-card-title {{cy.$id(focusId).data('repr')}}
      v-card-text(style="flex-grow: 1; flex-shrink: 1; overflow-y: auto")
        table.attrs
          tr(v-for="[k, v] of Object.entries(cy.$id(focusId).data('attrs') || {})")
            td.key {{k}}
            td.value {{v}}
          tr(v-for="obj of cy.$id(focusId).connectedEdges().toArray()")
            td.key
              //- Two equal for str compare
              span(v-if="obj.data('source') == focusId") obs_as_src
              span(v-else) obs_as_dst
            td.value
              span {{obj.data('repr')}}
          //- Entity shows fusions
          tr(v-if="!cy.$id(focusId).data('target')" v-for="fuse of cy.$id(focusId).data('fusions')")
            td.key fusion
            td.value {{fuse.id_other}} {{fuse.comment}}
      v-card-actions
        v-btn(color="primary" @click="focusDetails = false") Close

  div.tooltips(:refresh="cyUpdateNumber")
    div(v-for="id of cyTooltips.keys()" :key="id" :style="tooltipStyle(id)")
      template(v-if="cy.$id(id).data('target')")
        <!-- observation -->
        span {{cy.$id(id).data('repr')}} -- {{dbTimeToDisplay(cy.$id(id).data('time'))}}
      template(v-else)
        <!-- entity -->
        span {{cy.$id(id).data('repr')}}
      div(v-for="[k, v] of Object.entries(cy.$id(id).data('attrs') || {})") {{k}}: {{new String(v).length > 100 ? new String(v).substring(0, 100) + '...' : new String(v) + ''}}

  div(style="position: absolute; left: 2em; right: 2em; bottom: 0.5em")
    div(style="display: flex; flex-direction: row; align-items: center; justify-items: center;")
      //- Fix width/height so loading/not doesn't cause UI jitter
      div(style="padding: 0 1em; width: 3.5em; height: 2em")
        span(v-if="loadSetCurrent !== loadSetComplete")
          v-icon.blink mdi-clock
        v-btn(v-else icon variant="outlined" size="x-small"
            @click="cyIsRunning=!cyIsRunning")
          v-icon(v-if="cyIsRunning") mdi-pause
          v-icon(v-else) mdi-play
      span(style="white-space: nowrap; width: 4em;") {{cy !== undefined ? cy.$().length : 'loading'}} cyObjs
      span(style="padding-left: 1em") Timeline
      span.textbtn(@click="timeMeanModel -= timeRange") &lt;&lt;
      span.textbtn(@click="timeMeanModel -= timeRange * 0.2") &lt;
      span(style="white-space: nowrap")
        select(v-model="timeMeanYear")
          option(v-for="year of timeMeanYearOptions" :value="year") {{year}}
        span -
        select(v-model="timeMeanMonth")
          option(v-for="month of timeMeanMonthOptions" :value="month") {{month+1}}
        span -
        select(v-model="timeMeanDay")
          option(v-for="day of timeMeanDayOptions" :value="day") {{day}}
        span= " +/- "
        select(v-model="timeRangeModel")
          option(v-for="r of timeRangeOptions" :value="r.value") {{r.label}}
      span.textbtn(@click="timeMeanModel += timeRange * 0.2") &gt;
      span.textbtn(@click="timeMeanModel += timeRange") &gt;&gt;
      span Hops
      span
        select(v-model="graphHopsModel")
          option(v-for="k of graphHopsOptions" :value="k") {{k}}
      span(style="flex-grow: 1")
        vuespa-searchbar(v-model="focusIdModel" endpoint-data="entity_search"
            endpoint-label="entity_search_reverse"
            label="Entity search")
      //- TODO:
        span cluster files based on directory structure...
</template>
<style lang="scss">

.v-overlay__content {
  max-width: 90vw !important;
  max-height: 80vh !important;
}

table.attrs {
  tr ~ tr td {
    border-top: solid 2px #ccc;
  }
  td {
    vertical-align: top;
    white-space: pre-wrap;
    word-break: break-all;

    &.key {
      min-width: 4em;
    }
  }
}

.temporalview {
  position: relative;
  height: 100vh;
  width: 100vw;

  .blink {
    animation: blink 1s linear infinite;
  }
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }

  .cytoscape {
    height: 100%;
    width: 100%;
    text-align: left;
  }

  .textbtn {
    padding: 0 0.5em;
    cursor: pointer;
    user-select: none;
  }

  .tooltips {
    pointer-events: none;

    > div {
      position: absolute;
      text-align: center;
      background-color: #000;
      border-radius: 0.25em;
      padding: 0.25em;
      color: #fff;
    }
  }

  select {
    margin: 0 0.25em;
    padding: 0 0.25em;
  }
}
</style>
<script lang="ts">
import VuespaSearchbar from '@/components/VuespaSearchbar.vue';

import colormap from 'colormap';
import cytoscape from 'cytoscape';
import cola from 'cytoscape-cola';
import dayjs from 'dayjs';
import {defineComponent} from 'vue';

cytoscape.use(cola);

export default defineComponent({
  components: {
    VuespaSearchbar,
  },
  props: {
    focusId: {type: Number, required: true},
    graphHops: {type: Number, required: true},
    timeMean: {type: Number, required: true},
    timeRange: {type: Number, required: true},
  },
  data() {
    return {
      colors: colormap({
        colormap: 'viridis',
        nshades: 200,
        format: 'hex',
        alpha: 1,
      }),
      cyIsRunning: true,
      cyIsRunningLayout: null as any,
      cyTooltips: new Map<any, any>(),
      cyUpdateHandle: null as any,
      cyUpdateNumber: 0, // Hack for pan to update tooltips
      dayjs,
      focusDetails: false,
      loadSetCurrent: 0,
      loadSetComplete: 0,
      mountedFlag: false,
    };
  },
  computed: {
    cy(): any { // cytoscape.Core {
      // Weird $options access workaround
      if (!this.mountedFlag) return undefined as any as cytoscape.Core;
      return this.$options.cy as cytoscape.Core;
    },
    focusIdModel: {
      get(): number { return this.focusId; },
      set(v: number) {
        this.$emit('update:focusId', v);
      },
    },
    graphHopsModel: {
      get(): number { return this.graphHops; },
      set(v: number) {
        if (this.isLoading) return;
        this.$emit('update:graphHops', v);
      },
    },
    graphHopsOptions(): Array<number> {
      return [...Array(6).keys()];
    },
    isLoading(): boolean {
      return this.loadSetCurrent !== this.loadSetComplete;
    },
    timeMeanDay: {
      get(): number {
        return dayjs(this.timeMean).date();
      },
      set(v: any) {
        this.timeMeanModel = +dayjs(this.timeMean).date(v);
      },
    },
    timeMeanDayOptions(): Array<number> {
      const dim = dayjs(this.timeMean).daysInMonth();
      return [...Array(dim).keys()].map(x => x+1);
    },
    timeMeanModel: {
      get(): number { return this.timeMean; },
      set(v: number) {
        if (this.isLoading) return;
        this.$emit('update:timeMean', v);
      },
    },
    timeMeanMonth: {
      get(): number {
        return dayjs(this.timeMean).month();
      },
      set(v: any) {
        this.timeMeanModel = +dayjs(this.timeMean).month(v);
      },
    },
    timeMeanMonthOptions(): Array<number> {
      return [...Array(12).keys()];
    },
    timeMeanYear: {
      get(): number {
        return dayjs(this.timeMean).year();
      },
      set(v: any) {
        this.timeMeanModel = +dayjs(this.timeMean).year(v);
      },
    },
    timeMeanYearOptions(): Array<number> {
      const y = dayjs(this.timeMean).year();
      return [...Array(20).keys()].map(x => x + y - 10);
    },
    timeRangeModel: {
      get(): number { return this.timeRange; },
      set(v: number) {
        if (this.isLoading) return;
        this.$emit('update:timeRange', v);
      },
    },
    timeRangeOptions(): Array<any> {
      const d = 86400000;
      return [
        {label: '1d', value: d},
        {label: '1w', value: d*7},
        {label: '1mo', value: d*30},
        {label: '1y', value: d*365.25},
        {label: '5y', value: d*365.25*5},
      ];
    },
  },
  watch: {
    cyIsRunning() {
      // Newly running or newly stopping?
      this.cyLayoutRestart();
    },
    focusId() {
      (async () => {
        const focusId = this.focusId;
        if (focusId === null) return;

        let loadNew = !this.cyGetEntity(focusId).length;
        if (loadNew) {
          // Reset other elements
          this.cy.$().remove();
          this.cyTooltips.clear();
        }

        if (focusId === null) return;

        this.cy.center(this.cyGetEntity(focusId));
        if (loadNew) {
          await this.ensureEntity(focusId);
          this.cyLayoutRestart();
        }
        await this.cyUpdateGraph();
      })().catch(console.error);
    },
    graphHops() {
      this.cyUpdateGraph().catch(console.error);
    },
    timeMean() {
      this.cyUpdateGraph().catch(console.error);
    },
    timeRange() {
      this.cyUpdateGraph().catch(console.error);
    },
  },
  mounted() {
    // Fix options
    if (this.focusId < 0) {
      (async () => {
        const e = await this.$vuespa.call('entity_random');
        this.$emit('update:focusId', e.id);
      })().catch(console.error);
    }
    else {
      // Must trigger load
      setTimeout(() => {
        (async () => {
          const focusId = this.focusId;
          await this.ensureEntity(focusId);
          await this.cyUpdateGraph();
        })().catch(console.error);
      }, 0);
    }
    if (this.timeMean < 0) {
      this.$emit('update:timeMean', Date.now());
    }
    if (this.timeRange < 0) {
      this.$emit('update:timeRange', 86400000 * 365.25);
    }
    if (this.graphHops < 0) {
      this.graphHopsModel = 3;
    }

    // use $options to avoid reactivity
    this.$options.cy = cytoscape({
      container: this.$refs.cytoscape as any,
      elements: [],
      style: [ // the stylesheet for the graph
        {
          selector: 'node',
          style: {
            //'label': 'data(repr)'
            'background-color': '#666',
          }
        },

        {
          selector: 'node[?focus]',
          style: {
            label: '<focus>',
            color: '#fff',
            'text-background-color': '#000',
            'text-background-opacity': 0.8,
            'text-background-padding': '0.1em',
          },
        },

        {
          selector: 'node[type="file"]',
          style: {
            shape: 'triangle',
            'background-color': '#F7F5FB',
            'border-color': '#111',
            'border-width': 1,
            'border-style': 'solid',
          },
        },

        {
          selector: 'node[type="git_commit"]',
          style: {
            shape: 'roundrectangle',
            'background-color': '#084887',
          },
        },

        {
          selector: 'node[type="person"]',
          style: {
            shape: 'rectangle',
            'background-color': '#F58A07',
          },
        },

        {
          selector: 'node[type="message"]',
          style: {
            shape: 'diamond',
            'background-color': '#4040E0',
          },
        },

        {
          selector: 'edge',
          style: {
            'width': 3,
            //'line-color': 'data(time)', // '#ccc',
            'line-color': this.lineColorByTime,
            'target-arrow-color': '#ccc',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier'
          }
        },

        {
          selector: 'node[?loadLimited]',
          style: {
            'label': 'data(repr)',
            'border-color': '#f00',
            'border-width': 2,
            'border-style': 'solid',
          },
        },

        {
          selector: 'node[?isTime]',
          style: {
            shape: 'star',
            'background-color': 'data(hintColor)',
          },
        },
      ],
    });

    this.mountedFlag = true;

    // Configure events
    this.cy.on('position pan zoom resize', () => {this.cyWasUpdated();});
    this.cy.on('tap', 'node', (e: any) => {
      const id = e.target.id();
      if (this.cyIsEntity(id)) {
        const eId = +id;
        if (this.focusId === eId) {
          this.focusDetails = true;
        }
        else {
          this.$emit('update:focusId', eId);
        }
      }
      else if (this.cyIsLoadLimited(id)) {
        (async () => {
          await this.loadEntityObservations(this.cy.$id(id).data('loadId'),
            {forceType: this.cy.$id(id).data('loadType')});
          await this.cyUpdateGraph();
        })().catch(console.error);
      }
      else if (this.cyIsTime(id)) {
        this.$emit('update:timeMean', +dayjs.unix(e.target.data('time')));
      }
    });
    this.cy.on('mouseover', 'node,edge', (e: any) => {
      const id = e.target.id();
      this.cyTooltips.has(id) || this.cyTooltips.set(id, {});
    });
    this.cy.on('mouseout', 'node,edge', (e: any) => {
      const id = e.target.id();
      if (this.cyTooltips.get(id)?.hold) {
        return;
      }
      this.cyTooltips.delete(id);
    });
  },
  methods: {
    /** Runs `this.cy.$id(id)`, adapting `id` for entities. */
    cyGetEntity(id: number) {
      return this.cy.$id(this.cyIdEntity(id));
    },
    /** Runs `this.cy.$id(id)`, adapting `id` for observations. */
    cyGetObservation(id: number) {
      return this.cy.$id(this.cyIdObservation(id));
    },
    cyIdEntity(id: number): string {
      return id.toString();
    },
    cyIdToEntity(cyId: string): number {
      if (!/^\d/.test(cyId)) throw new Error(`Not entity id: ${cyId}`);
      return +cyId;
    },
    cyIdObservation(id: number): string {
      return `o${id}`;
    },
    cyIdToObservation(cyId: string): number {
      if (!cyId.startsWith('o')) throw new Error(`Not obs id: ${cyId}`);
      return +cyId.substring(1)
    },
    cyIsEntity(cyId: string): boolean {
      if (/^\d/.test(cyId)) return true;
      return false;
    },
    cyIsLoadLimited(cyId: string): boolean {
      if (/^tmp/.test(cyId)) return true;
      return false;
    },
    cyIsObservation(cyId: string): boolean {
      return cyId.startsWith('o');
    },
    cyIsTime(cyId: string): boolean {
      return /^time/.test(cyId);
    },
    cyLayoutRestart() {
      this.cyIsRunningLayout?.stop();
      if (!this.cyIsRunning) return;
      this.cyIsRunningLayout = this.cy.layout(this.layoutGetParams()).run();
    },
    cyWasUpdated() {
      if (this.cyUpdateHandle !== null) return;
      this.cyUpdateHandle = setTimeout(() => {
        this.cyUpdateNumber++;
        this.cyUpdateHandle = null;
      }, 100);
    },
    /** Re-load all nodes around `focusId`, keeping only those matching the
      current viewing criteria.
      */
    async cyUpdateGraph() {
      // Filter out existing edges which are no longer in the temporal range
      const ts = (this.timeMean - this.timeRange) / 1000;
      const te = (this.timeMean + this.timeRange) / 1000;
      this.cy.$('edge').forEach((el: cytoscape.CollectionReturnValue) => {
        const et = el.data('time');
        if (ts > et || et > te) el.remove();
      });

      const focusId = this.focusId;
      if (focusId < 0) {
        // Nothing to do...
        return;
      }

      // Find hops to load for each node accessible from focusId
      let stack = this.cy.$id(focusId);
      this.cy.$().data('focus', false);
      stack.data('focus', true);
      for (let i = 0, m = this.graphHops+1; i < m; i++) {
        stack = stack.union(stack.neighborhood());
      }
      stack = stack.filter(
          (el: cytoscape.CollectionReturnValue) =>
            this.cyIsEntity(el.id()) || this.cyIsObservation(el.id()));
      this.cy.nodes().not(stack).remove();

      // Remove all tooltips which no longer apply
      for (const k of this.cyTooltips.keys()) {
        if (!this.cy.$id(k).length) {
          this.cyTooltips.delete(k);
        }
      }

      // Starting with the root node, do a BFS for connected nodes
      this.loadSetCurrent++;
      const loadSet = this.loadSetCurrent;
      const searched = new Set<string>();
      stack = this.cyGetEntity(focusId);
      for (let i = 0, m = this.graphHops+1; i < m; i++) {
        // `stack` are loaded, but not their observations
        const promises = new Array<Promise<void>>();
        stack.forEach((el: cytoscape.CollectionReturnValue) => {
          const id = el.id();
          if (!this.cyIsEntity(id)) return;
          if (searched.has(id)) return;
          searched.add(id);
          promises.push(this.loadEntityObservations(this.cyIdToEntity(id),
              {loadSet}));
        });
        await Promise.all(promises);
        if (loadSet !== this.loadSetCurrent) return;

        // Expand next search step, since we've loaded connections
        stack = stack.union(stack.neighborhood('node'));
      }

      if (loadSet !== this.loadSetCurrent) return;
      this.loadSetComplete = loadSet;

      // Find preceding / following events
      const [before, after] = await this.$vuespa.call('entity_obs_adjacent',
          focusId, ts, te);
      if (before !== null) {
        let repr = `Next earliest event: ${this.dbTimeToDisplay(before.time)}`;
        this.cy.add([
            {data: {id: 'time-eventBefore',
              repr: repr,
              isTime: true,
              time: before.time,
              hintColor: this.colors[0]}},
            {data: {source: this.cyIdEntity(this.focusId), target: 'time-eventBefore',
              repr: repr, time: before.time}},
        ]);
      }
      if (after !== null) {
        let repr = `Next latest event: ${this.dbTimeToDisplay(after.time)}`;
        this.cy.add([
            {data: {id: 'time-eventAfter',
              repr: repr,
              isTime: true,
              time: after.time,
              hintColor: this.colors[this.colors.length-1]}},
            {data: {source: this.cyIdEntity(this.focusId), target: 'time-eventAfter',
              repr: repr, time: after.time}},
        ]);
      }

      // Re-color existing edges and re-layout
      this.cyUpdateStyles();
      this.cyLayoutRestart();
    },
    cyUpdateStyles() {
      this.cy.edges().forEach((el: any) => {
        el.style({'line-color': this.lineColorByTime(el)});
      });
    },
    dbTimeToDisplay(v: number) {
      return dayjs.unix(v).format('YYYY-MM-DD');
    },
    async ensureEntity(id: number, opts: {from?: any, e?: any}={}) {
      if (this.cyGetEntity(id).length) return;

      let e = opts.e;
      if (e === undefined) {
        e = await this.$vuespa.call('entity_get', id);
      }

      const node: any = {data: e};
      if (opts.from !== undefined) {
        const el = this.cyGetEntity(opts.from);
        if (el.length) {
          node.position = Object.assign({}, el.position());
          node.position.x += (Math.random() - 0.5) * 50;
          node.position.y += (Math.random() - 0.5) * 50;
        }
      }

      this.cy.add([node]);
      return e;
    },
    async ensureObservation(id: number, o?: any) {
      if (this.cyGetObservation(id).length) return;

      if (o === undefined) {
        o = await this.$vuespa.call('observation_get', id);
      }

      o.id = this.cyIdObservation(id);
      o.source = o.src_id;
      o.target = o.dst_id;
      this.cy.add([{data: o}]);
      return o;
    },
    layoutGetParams() {
      // Cola seems to work best
      return {
        name: 'cola',
        fit: false,
        maxSimulationTime: 1e4,
        infinite: true,
        refresh: 2,  // sim steps per frame
        randomize: false,

        edgeLength: (el: cytoscape.CollectionReturnValue) => {
          if (el.data('loadLimitedEdge')) return 20;
          return 200;
        },
      };

      // But here is an example of an fcose setup for comparison
      return {
        name: 'fcose',
        fit: false,
        //These two go together
        randomize: false, quality: 'proof',
        numIter: null,

        nodeRepulsion: 45000,
        idealEdgeLength: (el: cytoscape.CollectionReturnValue) => {
          if (el.data('loadLimitedEdge')) return 20;
          return 200;
        },
        edgeElasticity: 20,
        //gravity: 0.1,

        //samplingType: false,
        sampleSize: 50,
        //nodeSeparation: 750,
      };
    },
    lineColorByTime(el: any) {
      const v = el.data('time') * 1000;
      let u = (v - this.timeMean) / this.timeRange;
      u = Math.max(-1, Math.min(1, u));
      u *= 0.5;
      u += 0.5;
      return this.colors[Math.floor(u * (this.colors.length - 0.01))];
    },
    async loadEntityObservations(id: number, opts: {loadSet?: number,
        forceType?: string}) {
      const ts = (this.timeMean - this.timeRange) / 1000;
      const te = (this.timeMean + this.timeRange) / 1000;
      const foci: Array<any> = await this.$vuespa.call('entity_get_obs', id, ts, te);

      // Cytoscape isn't great at showing large numbers of things. So, group.
      // Note that we only consider growth rate here -- any already-loaded nodes
      // MUST be linked or else the visualization would be useless.
      const entCap = 5;
      interface EntTypeGroup {
        ents: Map<number, any>;
        timeMax: number;
        timeMin: number;
      }
      let entsByType = new Map<string, EntTypeGroup>();
      for (let f of foci) {
        let type = entsByType.get(f.type);
        if (type === undefined) {
          type = {ents: new Map<number, any>(),
              timeMax: Number.NEGATIVE_INFINITY,
              timeMin: Number.POSITIVE_INFINITY};
          entsByType.set(f.type, type);
        }

        if (f.src_id !== null && !this.cyGetEntity(f.src_id).length) {
          type.ents.set(f.src_id, {from: f.dst_id});
          type.timeMax = Math.max(type.timeMax, f.time);
          type.timeMin = Math.min(type.timeMin, f.time);
        }
        if (!this.cyGetEntity(f.dst_id).length) {
          type.ents.set(f.dst_id, {from: f.src_id});
          type.timeMax = Math.max(type.timeMax, f.time);
          type.timeMin = Math.min(type.timeMin, f.time);
        }
      }

      const promises = [];
      for (const [loadType, loadMap] of entsByType.entries()) {
        if (loadType !== opts.forceType && loadMap.ents.size > entCap) {
          // Add an ad hoc group here
          const cId = `tmp_${id}_${loadType}`;
          this.cy.add([
            {data: {id: cId, loadLimited: true, repr: `${loadMap.ents.size} ${loadType}`,
              loadId: id, loadType}, position: {
                x: (Math.random() - 0.5) * 50 + this.cy.$id(id).position('x'),
                y: (Math.random() - 0.5) * 50 + this.cy.$id(id).position('y'),
                }},
            {data: {source: id, target: cId, time: loadMap.timeMin,
              loadLimitedEdge: true, repr: loadType}},
            {data: {source: id, target: cId, time: loadMap.timeMax,
              loadLimitedEdge: true, repr: loadType}},
          ]);
        }
        else {
          promises.push(...Array.from(loadMap.ents.entries(), ([a, b]) => this.ensureEntity(a, b)));
        }
      }
      promises.length && (await Promise.all(promises));
      if (opts.loadSet !== undefined && opts.loadSet !== this.loadSetCurrent) return;

      // Finally, add remaining observations, excepting those which weren't
      // loaded
      promises.length = 0;
      for (const f of foci) {
        if (f.src_id !== null && !this.cyGetEntity(f.src_id).length) continue;
        if (!this.cyGetEntity(f.dst_id).length) continue;
        promises.push(this.ensureObservation(f.id, f));
      }
      await Promise.all(promises);
    },
    tooltipStyle(id: any) {
      const pos = this.cy.$id(id).renderedBoundingBox();
      const x = (pos.x1 + pos.x2) * 0.5;
      const y = (pos.y1 + pos.y2) * 0.5;
      return {
        left: x + 'px',
        top: y + 'px',
      };
    },
  },
});
</script>


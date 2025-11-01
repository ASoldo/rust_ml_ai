pub const HUD_ATAK_HTML: &str = r#"
<!doctype html>
<html lang="en" class="h-full dark">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ATAK Mission Console</title>

  <!-- Fonts & UI -->
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css"
    crossorigin="anonymous" referrerpolicy="no-referrer" />
  <style type="text/tailwindcss">
    @theme {
      /* Fonts */
      --font-tactical: "Rajdhani", ui-sans-serif, system-ui;

      /* Colors */
      --color-mission-bg: #050B16;
      --color-mission-panel: #0A1628;
      --color-mission-panel-light: #13233D;
      --color-mission-accent: #38bdf8;
      --color-mission-warning: #facc15;
      --color-mission-danger: #fb7185;
      --color-mission-success: #4ade80;

      /* Shadows */
      --shadow-hud: 0 0 24px rgba(56, 189, 248, 0.18);
      --shadow-hud-soft: 0 0 18px rgba(15, 23, 42, 0.65);
    }

    @layer base {
      html { @apply h-full; }
      body { @apply h-full bg-mission-bg text-slate-100 font-[Rajdhani]; }
      [x-cloak] { display: none !important; }
      #map { @apply absolute inset-0 z-0 h-full w-full; }

    }

    @layer components {
      .dot-marker {
        width: 18px;
        height: 18px;
        border-radius: 50%;
        background: rgba(56, 189, 248, 0.95);
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.4), 0 0 12px rgba(56, 189, 248, 0.55);
        border: 2px solid rgba(15, 23, 42, 0.85);
      }

      .draft-marker {
        @apply flex h-[22px] w-[22px] items-center justify-center rounded-full border text-[11px] font-bold;
        background: rgba(56, 189, 248, 0.92);
        border-color: rgba(248, 250, 252, 0.55);
        color: #0f172a;
        box-shadow: 0 0 12px rgba(15, 23, 42, 0.65);
      }

      .connector-arrow {
        width: 0;
        height: 0;
        border-left: 7px solid transparent;
        border-right: 7px solid transparent;
        border-bottom: 14px solid #38bdf8;
        filter: drop-shadow(0 0 6px rgba(56, 189, 248, 0.45));
        transform-origin: 50% 60%;
      }

      .marker-cluster-small div {
        background: rgba(56, 189, 248, 0.85) !important;
        color: #0b1222 !important;
      }

      .marker-cluster-medium div {
        background: rgba(56, 189, 248, 0.9) !important;
        color: #0b1222 !important;
      }

      .marker-cluster-large div {
        background: rgba(56, 189, 248, 1) !important;
        color: #0b1222 !important;
      }

      dialog.hud-dialog {
        @apply flex flex-col overflow-hidden;
        padding: 0;
        border: 1px solid rgba(56, 189, 248, 0.35);
        background: rgba(19, 35, 61, 0.96);
        border-radius: 18px;
        color: #e2e8f0;
        box-shadow: 0 22px 45px rgba(0, 0, 0, 0.55);
        max-height: calc(100vh - env(safe-area-inset-top, 0px) - env(safe-area-inset-bottom, 0px) - 1rem);
      }

      dialog.hud-dialog::backdrop {
        background: rgba(5, 11, 22, 0.85);
        backdrop-filter: blur(6px);
      }

      dialog.layers-sheet {
        position: fixed;
        left: 50%;
        bottom: calc(env(safe-area-inset-bottom, 0px));
        transform: translate(-50%, 105%);
        transition: transform 0.28s ease, opacity 0.28s ease;
        width: min(520px, calc(100vw - 1.25rem));
        height: calc(var(--sheet-height, 0.5) * 100vh);
        max-height: calc(0.85 * 100vh);
        margin: 0;
        border-radius: 18px 18px 0 0;
        overflow: hidden;
        opacity: 0;
        pointer-events: none;
      }

      dialog.layers-sheet[open] {
        transform: translate(-50%, 0);
        opacity: 1;
        pointer-events: auto;
        box-shadow:
          0 -22px 38px rgba(5, 11, 22, 0.6),
          0 -6px 18px rgba(56, 189, 248, 0.18);
      }

      dialog.filter-dropdown {
        position: absolute;
        margin: 0;
        border-radius: 12px;
        padding: 0;
        border: 1px solid rgba(56, 189, 248, 0.35);
        background: rgba(19, 35, 61, 0.96);
        color: #e2e8f0;
        box-shadow:
          0 18px 32px rgba(5, 11, 22, 0.55),
          0 0 18px rgba(56, 189, 248, 0.12);
        min-width: 160px;
        display: none;
        z-index: 80;
      }

      dialog.overlap-dialog {
        z-index: 90;
      }

      dialog.filter-dropdown[open] {
        display: block;
      }

      .filter-option {
        width: calc(100% - 12px);
        margin: 6px;
        text-align: left;
        padding: 0.6rem 0.9rem;
        font-size: 12px;
        letter-spacing: 0.25em;
        text-transform: uppercase;
        color: #e2e8f0;
        background: rgba(19, 35, 61, 0.6);
        border: none;
        border-radius: 10px;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        transition: background 0.2s ease, color 0.2s ease;
      }

      .filter-option:hover,
      .filter-option:focus {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        outline: none;
      }

      .sheet-handle {
        position: relative;
        width: 100%;
        padding: 0.75rem 0;
        display: flex;
        justify-content: center;
        cursor: grab;
        user-select: none;
        touch-action: none;
      }

      .sheet-handle::after {
        content: '';
        width: 54px;
        height: 5px;
        border-radius: 999px;
        background: rgba(148, 163, 184, 0.45);
      }

      .sheet-handle:active {
        cursor: grabbing;
      }

      @media (max-width: 640px) {
        dialog.layers-sheet {
          width: calc(100vw - 0.75rem);
        }
      }
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>

  <!-- Leaflet 1.9.4 -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css" crossorigin="anonymous"
    referrerpolicy="no-referrer" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js" crossorigin="anonymous"
    referrerpolicy="no-referrer"></script>

  <!-- Leaflet.markercluster 1.4.1 -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.4.1/dist/MarkerCluster.Default.css" />
  <script src="https://unpkg.com/leaflet.markercluster@1.4.1/dist/leaflet.markercluster.js"></script>

  <style>
    .leaflet-container {
      background: #050b16;
    }

    .leaflet-popup-content-wrapper {
      background: rgba(15, 23, 42, .95);
      color: #f8fafc;
      border-radius: 10px;
      border: 1px solid rgba(56, 189, 248, .35);
      box-shadow: 0 18px 30px rgba(0, 0, 0, .45);
    }

    .leaflet-popup-tip {
      background: rgba(15, 23, 42, .95);
    }

    .leaflet-control {
      border: none !important;
      border-radius: 8px;
      box-shadow: 0 0 18px rgba(15, 23, 42, .45);
      overflow: hidden;
    }

    .leaflet-bar a {
      background: rgba(15, 23, 42, .92);
      border-bottom: 1px solid rgba(56, 189, 248, .2);
      color: #e2e8f0;
    }

    .leaflet-bar a:hover,
    .leaflet-bar a:focus {
      background: rgba(56, 189, 248, .2);
      color: #38bdf8;
    }
  </style>

  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.13.5/dist/cdn.min.js" crossorigin="anonymous"></script>
</head>

<body x-data="atakApp()" x-init="init()" class="bg-mission-bg text-slate-100 h-full font-[Rajdhani]">
  <div class="flex flex-col h-screen">
    <!-- Header -->
    <header
      class="relative z-30 border-b border-slate-800/70 bg-mission-panel/90 px-3 py-2 md:px-4 md:py-2.5 shadow-hud-soft">
      <div class="flex flex-wrap items-center justify-between gap-3 md:gap-6">
        <div class="min-w-0">
          <p class="text-[9px] md:text-[10px] uppercase tracking-[0.35em] md:tracking-[0.4em] text-slate-500">ATAK
            Network Dashboard</p>
          <h1 class="truncate text-base md:text-xl font-semibold text-mission-accent" x-text="currentOperation"></h1>
        </div>
        <div class="flex items-center gap-3 md:gap-4 text-[11px] md:text-xs">
          <div class="text-right">
            <p class="uppercase text-[9px] md:text-[11px] tracking-[0.25em] md:tracking-[0.3em] text-slate-500">Zulu</p>
            <p class="font-semibold leading-none md:text-base" x-text="localTime"></p>
          </div>
          <div class="pl-3 md:pl-6 md:border-l md:border-slate-700/70">
            <p class="uppercase text-[9px] md:text-[11px] tracking-[0.25em] md:tracking-[0.3em] text-slate-500">Callsign
            </p>
            <p class="font-semibold leading-none md:text-base" x-text="missionCallsign"></p>
          </div>
          <div class="pl-3 md:pl-6 md:border-l md:border-slate-700/70">
            <p class="uppercase text-[9px] md:text-[11px] tracking-[0.25em] md:tracking-[0.3em] text-slate-500">Net</p>
            <p :class="['font-semibold leading-none md:text-base', networkStatusClass]" x-text="networkStatusLabel"></p>
          </div>
        </div>
      </div>
    </header>

    <main class="relative flex flex-1 overflow-hidden bg-mission-bg min-h-0">
      <section class="relative flex-1 overflow-hidden bg-mission-bg min-h-0">
        <div id="map"></div>

        <!-- Over-map panels (unchanged UI) -->
        <div class="pointer-events-none absolute top-2 md:top-4 left-0 right-0 z-20 px-2 md:px-4">
          <div class="relative flex items-start justify-center">
            <div
              class="pointer-events-auto flex w-full flex-col items-center gap-2 rounded-none border border-transparent bg-transparent px-1.5 py-1.5 shadow-none md:w-auto md:px-2 md:py-2">
              <div class="flex flex-wrap items-center justify-center gap-1.5">
                <button type="button"
                  class="flex h-8 items-center gap-1.5 rounded-lg border border-slate-700/70 px-2.5 text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-100 hover:border-mission-accent/60 hover:text-mission-accent md:h-9 md:gap-2 md:px-3 md:text-xs md:tracking-[0.2em]"
                  :class="mode === 'pan' ? 'border-mission-accent/70 bg-mission-accent/20 text-mission-accent shadow-hud ring-1 ring-mission-accent/40' : ''"
                  title="Pan / Navigate" @click="setMode('pan')">
                  <i class="fa-solid fa-up-down-left-right text-xs md:text-sm"></i>
                  <span class="hidden md:inline">Pan</span>
                  <span class="hidden md:inline text-[10px] uppercase tracking-[0.3em]">ESC</span>
                </button>
              <div class="relative flex flex-col items-center">
                  <button type="button" x-ref="toolButton"
                  class="flex h-8 items-center gap-1.5 rounded-lg border border-slate-700/70 px-2.5 text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-100 hover:border-mission-accent/60 hover:text-mission-accent md:h-9 md:gap-2 md:px-3 md:text-xs md:tracking-[0.2em]"
                  :class="selectedTool ? 'border-mission-accent/70 bg-mission-accent/15 text-mission-accent shadow-hud ring-1 ring-mission-accent/40' : ''"
                  aria-haspopup="listbox"
                  :aria-expanded="toolDropdownOpen ? 'true' : 'false'"
                  @click.stop="toggleToolDropdown($event)">
                  <i class="fa-solid" :class="currentToolOption().icon"></i>
                  <span class="hidden md:inline" x-text="currentToolOption().label"></span>
                </button>
                <dialog x-ref="toolDialog" class="filter-dropdown"
                  x-init="
                    const dialog = $refs.toolDialog;
                    const ctx = $data;
                    const close = () => { if (dialog.open) dialog.close(); };
                    const open = () => { if (!dialog.open) dialog.show(); };
                    $watch('toolDropdownOpen', value => { value ? open() : close(); });
                    dialog.addEventListener('click', (event) => {
                      if (event.target === dialog) ctx.closeToolDropdown();
                    });
                    dialog.addEventListener('cancel', (event) => {
                      event.preventDefault();
                      ctx.closeToolDropdown();
                    });
                  "
                  @close="toolDropdownOpen = false">
                  <div class="py-1">
                    <template x-for="tool in toolOptions" :key="tool.value">
                      <button type="button" class="filter-option"
                        :class="selectedTool === tool.value ? 'text-mission-accent bg-mission-panel/60' : ''"
                        @click="setTool(tool.value)">
                        <i class="fa-solid" :class="tool.icon"></i>
                        <span x-text="tool.label"></span>
                      </button>
                    </template>
                  </div>
                </dialog>
              <template x-if="currentDraft.type && ['polyline','polygon'].includes(currentDraft.type) && currentDraft.points.length > 0">
                <div class="pointer-events-auto absolute left-1/2 top-full flex -translate-x-1/2 translate-y-2 gap-3 md:gap-4">
                  <button type="button"
                    class="flex items-center gap-1.5 rounded-lg border border-mission-success/60 bg-mission-success/15 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.3em] text-mission-success hover:bg-mission-success/25 md:text-xs"
                    @click="finishActiveTool()">
                    <i class="fa-solid fa-check"></i>
                  </button>
                    <button type="button"
                      class="flex items-center gap-1.5 rounded-lg border border-mission-danger/60 bg-mission-danger/15 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.3em] text-mission-danger hover:bg-mission-danger/25 md:text-xs"
                      @click="cancelActiveTool()">
                      <i class="fa-solid fa-xmark"></i>
                    </button>
                </div>
              </template>
              </div>
              <button type="button"
                class="relative flex h-8 items-center gap-1.5 rounded-lg border border-slate-700/70 px-2.5 text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-100 hover:border-mission-accent/60 hover:text-mission-accent md:h-9 md:gap-2 md:px-3 md:text-xs md:tracking-[0.2em]"
                :class="layersOpen ? 'border-mission-accent/70 bg-mission-accent/20 text-mission-accent shadow-hud ring-1 ring-mission-accent/40' : ''"
                aria-haspopup="dialog"
                :aria-expanded="layersOpen ? 'true' : 'false'"
                @click="toggleLayers()">
                <i class="fa-solid fa-layer-group text-xs md:text-sm"></i>
                <span class="hidden md:inline">Layers</span>
                <span
                  class="absolute -right-2 -top-2 rounded-full border border-mission-accent/40 bg-mission-accent/80 px-1.5 py-0.5 text-[9px] font-semibold leading-none text-slate-950"
                  x-text="overlays.length"></span>
              </button>
            </div>

          </div>
        </div>

        <dialog x-ref="overlapDialog" class="pointer-events-auto filter-dropdown overlap-dialog"
          :style="overlapChooser.style"
          x-init="
            const dialog = $refs.overlapDialog;
            const ctx = $data;
            const close = () => { if (dialog.open) dialog.close(); };
            const open = () => { if (!dialog.open) dialog.show(); };
            $watch('overlapChooser.open', value => { value ? open() : close(); });
            dialog.addEventListener('click', (event) => {
              if (event.target === dialog) ctx.closeOverlapChooser();
            });
            dialog.addEventListener('cancel', (event) => {
              event.preventDefault();
              ctx.closeOverlapChooser();
            });
          "
          @close="overlapChooser.open = false">
          <div class="py-1">
            <template x-for="item in overlapChooser.items" :key="item.id">
              <button type="button" class="filter-option"
                @click="selectOverlapOverlay(item.id)">
                <i class="fa-solid fa-layer-group"></i>
                <div class="flex flex-col text-left">
                  <span class="text-xs font-semibold" x-text="item.label"></span>
                  <span class="text-[10px] uppercase tracking-[0.25em] text-slate-400" x-text="item.type"></span>
                </div>
              </button>
            </template>
          </div>
        </dialog>

        <dialog x-ref="layersDialog" class="hud-dialog layers-sheet"
          x-init="
            const dialog = $refs.layersDialog;
            const ctx = $data;
            const syncHeight = () => { dialog.style.setProperty('--sheet-height', (sheetHeight || 0.5)); dialog.style.height = `calc(${sheetHeight || 0.5} * 100vh)`; };
            const closeDialog = () => { if (dialog.open) dialog.close(); };
            const openDialog = () => { if (!dialog.open) dialog.show(); };
            $watch('layersOpen', value => {
              if (value) ctx.closeToolDropdown();
              if (value) {
                openDialog();
                syncHeight();
              } else {
                finishSheetResize();
                closeDialog();
                ctx.cancelPendingDelete();
                ctx.closeFilterDialog();
                ctx.closeToolDropdown();
                ctx.closeOverlapChooser();
              }
            });
            $watch('sheetHeight', () => { if (dialog.open) syncHeight(); });
            syncHeight();
            dialog.addEventListener('click', (event) => { if (event.target === dialog) { ctx.layersOpen = false; }});
          "
          @cancel.prevent="layersOpen = false"
          @close="layersOpen = false">
          <div class="mobile-layers-content flex flex-1 min-h-0 w-full flex-col">
            <div class="sheet-handle" role="separator" aria-orientation="horizontal" aria-label="Resize layers panel" @pointerdown.prevent="beginSheetResize($event)"></div>
            <template x-if="editingLayerId || pendingDeleteId !== null">
              <div class="flex flex-wrap items-center justify-between gap-3 border-b border-slate-700/60 bg-mission-panel-light/85 px-4 py-2 text-xs text-slate-200">
                <template x-if="editingLayerId">
                  <div class="flex flex-1 items-center justify-between gap-3">
                    <span class="truncate font-semibold uppercase tracking-[0.25em] text-mission-warning" x-text="editingOverlaySummary()"></span>
                    <div class="flex items-center gap-2">
                      <button type="button"
                        class="rounded-lg bg-mission-warning/90 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.25em] text-slate-900 shadow"
                        @click="completeEditing()">Save</button>
                      <button type="button"
                        class="rounded-lg border border-slate-600/70 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.25em] text-slate-200 hover:border-mission-danger/70 hover:text-mission-danger"
                        @click="cancelEditing()">Cancel</button>
                    </div>
                  </div>
                </template>
                <template x-if="!editingLayerId && pendingDeleteId !== null">
                  <div class="flex flex-1 flex-col gap-2 md:flex-row md:items-center md:justify-between">
                    <span class="font-semibold uppercase tracking-[0.25em] text-mission-danger">Delete <span class="text-slate-100" x-text="pendingDeleteLabel"></span>?</span>
                    <div class="flex flex-1 items-center justify-end gap-3">
                      <label class="flex items-center gap-2 text-[11px] text-slate-200">
                        <input type="checkbox" class="h-4 w-4 rounded border border-slate-600 bg-transparent" x-model="deleteConfirmDontAsk">
                        <span class="uppercase tracking-[0.2em]">Don't ask again</span>
                      </label>
                      <button type="button"
                        class="rounded-lg border border-slate-600/70 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.25em] text-slate-200"
                        @click="cancelPendingDelete()">Cancel</button>
                      <button type="button"
                        class="rounded-lg bg-mission-danger/90 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.25em] text-slate-900 shadow"
                        @click="executePendingDelete()">Delete</button>
                    </div>
                  </div>
                </template>
              </div>
            </template>
            <header class="flex shrink-0 items-center justify-between gap-3 border-b border-slate-700/60 bg-mission-panel-light/85 px-4 py-3">
              <div>
                <p class="text-[10px] uppercase tracking-[0.3em] text-slate-400">Operational Layers</p>
                <p class="text-sm font-semibold text-slate-100" x-text="`${overlays.length} active`"></p>
              </div>
              <div class="relative flex items-center gap-3">
                <button type="button" x-ref="filterButton"
                  class="flex items-center gap-2 rounded-lg border border-slate-600/70 bg-mission-panel-light/60 px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.25em] text-slate-200 hover:border-mission-accent/60 focus:outline-none focus:ring-2 focus:ring-mission-accent/50"
                  aria-haspopup="listbox"
                  :aria-expanded="filterDialogOpen ? 'true' : 'false'"
                  @click.stop="toggleFilterDialog($event)">
                  <i class="fa-solid" :class="currentFilterOption().icon"></i>
                  <span x-text="currentFilterOption().label"></span>
                </button>
                <dialog x-ref="filterDialog" class="filter-dropdown"
                  x-init="
                    const dialog = $refs.filterDialog;
                    const ctx = $data;
                    const close = () => { if (dialog.open) dialog.close(); };
                    const open = () => { if (!dialog.open) dialog.show(); };
                    $watch('filterDialogOpen', value => { value ? open() : close(); });
                    dialog.addEventListener('click', (event) => {
                      if (event.target === dialog) ctx.closeFilterDialog();
                    });
                    dialog.addEventListener('cancel', (event) => {
                      event.preventDefault();
                      ctx.closeFilterDialog();
                    });
                  "
                  @close="filterDialogOpen = false">
                  <div class="py-1">
                    <template x-for="option in filterOptions" :key="option.value">
                      <button type="button" class="filter-option"
                        :class="layerFilter === option.value ? 'text-mission-accent bg-mission-panel/60' : ''"
                        @click="setLayerFilter(option.value)">
                        <i class="fa-solid" :class="option.icon"></i>
                        <span x-text="option.label"></span>
                      </button>
                    </template>
              </div>
            </dialog>
                <button type="button"
                  class="flex h-9 w-9 items-center justify-center rounded-full border border-slate-600/70 text-slate-300 hover:text-mission-accent"
                  @click="layersOpen = false">
                  <i class="fa-solid fa-xmark text-base"></i>
                </button>
              </div>
            </header>
            <div class="flex-1 min-h-0 overflow-y-auto px-4 py-3 space-y-3">
              <template x-if="overlays.length === 0">
                <p class="text-xs text-slate-400">No overlays deployed. Use drawing modes or drop a marker.</p>
              </template>
              <template x-for="overlay in filteredOverlays()" :key="overlay.id">
                <div
                  class="rounded-xl border border-slate-700/60 bg-slate-900/70 p-3 text-xs shadow-hud-soft"
                  :class="{
                    'border-mission-accent/70 shadow-hud': selectedLayerId === overlay.id && editingLayerId !== overlay.id,
                    'border-mission-warning/80 shadow-hud': editingLayerId === overlay.id
                  }">
                  <div class="flex items-start justify-between gap-2">
                    <div class="min-w-0">
                      <p class="truncate text-sm font-semibold text-slate-100" x-text="overlay.label"></p>
                      <p class="text-[10px] uppercase tracking-[0.2em] text-slate-400" x-text="overlay.type"></p>
                    </div>
                  </div>
                  <div class="mt-2 space-y-1 text-[11px] text-slate-400">
                    <template x-if="overlay.type === 'marker'">
                      <p x-text="`Lat ${formatLat(overlay.lat)} | Lng ${formatLng(overlay.lng)}`"></p>
                    </template>
                    <template x-if="overlay.type === 'polyline'">
                      <p x-text="`Length ${formatLength(overlay.lengthM)}`"></p>
                    </template>
                    <template x-if="overlay.type === 'polygon' || overlay.type === 'rectangle'">
                      <p x-text="`Area ${formatArea(overlay.areaSqM)}`"></p>
                    </template>
                    <template x-if="overlay.type === 'connector'">
                      <p x-text="`${findOverlayLabel(overlay.startId)} → ${findOverlayLabel(overlay.endId)}`"></p>
                      <p x-text="`Style ${overlay.style} | Arrow ${overlay.direction}`"></p>
                      <p x-text="`Span ${formatLength(overlay.lengthM)}`"></p>
                    </template>
                    <p class="text-[10px] text-slate-500"
                      x-text="new Date(overlay.created).toLocaleTimeString('en-GB', { hour12: false }) + 'Z'"></p>
                  </div>
                  <div class="mt-3 grid gap-2"
                    :class="['polyline','polygon','rectangle','marker'].includes(overlay.type) ? 'grid-cols-3' : 'grid-cols-2'">
                    <button type="button"
                      class="rounded-lg border border-slate-600/70 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-100"
                       :class="lockedOverlayId === overlay.id ? 'opacity-50 pointer-events-none' : ''"
                      :disabled="lockedOverlayId === overlay.id"
                      @click="handleLayerFocus(overlay.id)">Focus</button>
                    <template x-if="['polyline','polygon','rectangle','marker'].includes(overlay.type)">
                      <button type="button"
                        class="rounded-lg border border-slate-600/70 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-100 hover:border-mission-warning/80 hover:text-mission-warning"
                         :class="lockedOverlayId === overlay.id ? 'opacity-50 pointer-events-none' : ''"
                        :disabled="lockedOverlayId === overlay.id"
                        @click="handleLayerEdit(overlay.id)"
                        x-text="editingLayerId === overlay.id ? 'Editing' : 'Edit'"></button>
                    </template>
                    <button type="button"
                      class="rounded-lg border border-slate-600/70 px-2 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-100 hover:border-mission-danger/80 hover:text-mission-danger"
                       :class="lockedOverlayId === overlay.id ? 'opacity-50 pointer-events-none' : ''"
                      :disabled="lockedOverlayId === overlay.id"
                      @click="handleLayerDelete(overlay.id)">Del</button>
                  </div>
                </div>
              </template>
            </div>
          </div>
        </dialog>

      </section>
    </main>
    <footer class="border-t border-slate-800/70 bg-mission-panel/90 px-3 py-2 text-xs text-slate-300 md:px-6 md:py-3 md:text-sm">
      <div class="flex items-center gap-3 overflow-hidden">
        <span class="uppercase tracking-[0.32em] text-[8px] text-slate-500 md:text-[10px]">Status</span>
        <span class="font-semibold text-slate-100" x-text="status.message"></span>
        <span class="truncate text-slate-400" x-text="status.detail"></span>
      </div>
    </footer>
  </div>

  <script>
    function atakApp() {
      return {
        map: null,

        // Master group of everything (kept for export/toGeoJSON)
        drawnItems: null,

        // Logical groups (markers use ClusterGroup)
        groups: {
          markers: null,     // L.MarkerClusterGroup
          routes: null,      // L.FeatureGroup
          zones: null,       // L.FeatureGroup
          boxes: null,       // L.FeatureGroup
          connectors: null   // L.FeatureGroup
        },

        layerIndex: {},
        overlays: [],
        selectedLayerId: null,
        mode: 'pan',
        isMobile: false,
        layersOpen: false,
        layerFilter: 'all',
        filterDialogOpen: false,
        skipDeleteConfirm: false,
        deleteConfirmDontAsk: false,
        pendingDeleteId: null,
        pendingDeleteLabel: 'This overlay',
        editingSnapshot: null,
        mobileMediaQuery: null,
        sheetHeight: 0.5,
        sheetDragActive: false,
        sheetDragStartY: 0,
        sheetStartHeight: 0,
        sheetResizeHandlers: {move: null, up: null},
        networkStatusLabel: '—',
        networkStatusClass: 'text-slate-300',
        filterOptions: [
          {value: 'all', label: 'All', icon: 'fa-grip-lines'},
          {value: 'marker', label: 'Marker', icon: 'fa-location-dot'},
          {value: 'route', label: 'Route', icon: 'fa-route'},
          {value: 'zone', label: 'Zone', icon: 'fa-draw-polygon'},
          {value: 'box', label: 'Box', icon: 'fa-square'}
        ],
        toolOptions: [
          {value: 'marker', label: 'Marker', icon: 'fa-location-dot'},
          {value: 'polyline', label: 'Route', icon: 'fa-route'},
          {value: 'polygon', label: 'Zone', icon: 'fa-draw-polygon'},
          {value: 'rectangle', label: 'Box', icon: 'fa-square'}
        ],
        selectedTool: null,
        toolDropdownOpen: false,
        toolOutsideHandler: null,
        filterOutsideHandler: null,
        overlapChooser: {open: false, items: [], style: {top: '0px', left: '0px', minWidth: '160px'}},
        lockedOverlayId: null,
        status: {message: 'System initializing', detail: 'Stand by…'},
        markerForm: {label: '', symbol: '▲', color: '#38bdf8', persist: false, manualLat: '', manualLng: ''},
        missionCallsign: 'ROOTS-ACTUAL',
        currentOperation: 'OPERATION ARCTIC WATCH',
        localTime: '',
        cursorLat: null,
        cursorLng: null,
        zoomLevel: null,
        currentDraft: {type: null, points: [], layer: null, markers: []},
        keydownHandler: null,
        resizeHandler: null,
        connectorForm: {startId: '', endId: '', style: 'solid', direction: 'none', anchorMode: 'auto'},
        connectorArtifacts: {},
        editingLayerId: null,
        editingMarkers: [],
        editingData: null,

        async init() {
          this.initClock();
          this.setupResponsiveWatchers();
          this.applySheetHeight(this.sheetHeight);
          this.updateNetworkStatus();
          this.updateStatus('Initializing map stack', 'Loading Leaflet 1.9.4 with MarkerCluster…');
          try {
            await this.initMap();
            this.bindKeyControls();
            this.updateStatus('Console ready', 'Map secured. Awaiting tasking.');
          } catch (error) {
            console.error('ATAK initialization error', error);
            this.updateStatus('Map initialization failed', 'Leaflet library unavailable.');
          }
        },

        initClock() {
          const update = () => {
            const now = new Date();
            this.localTime = now.toLocaleTimeString('en-GB', {hour12: false}) + 'Z';
          };
          update();
          setInterval(update, 1000);
        },

        setupResponsiveWatchers() {
          if (typeof window === 'undefined' || !window.matchMedia) {
            this.isMobile = false;
            return;
          }

          if (this.mobileMediaQuery?.query && this.mobileMediaQuery.handler) {
            const {query, handler} = this.mobileMediaQuery;
            if (query.removeEventListener) query.removeEventListener('change', handler);
            else if (query.removeListener) query.removeListener(handler);
          }

          const syncState = (matches) => {
            this.isMobile = matches;
          };

          const mediaQuery = window.matchMedia('(max-width: 767px)');
          const handler = (event) => syncState(event.matches);
          syncState(mediaQuery.matches);
          if (mediaQuery.addEventListener) mediaQuery.addEventListener('change', handler);
          else if (mediaQuery.addListener) mediaQuery.addListener(handler);
          this.mobileMediaQuery = {query: mediaQuery, handler};
        },

        async initMap() {
          if (!window.L || typeof L.map !== 'function') throw new Error('Leaflet core unavailable');

          // Fresh map node (don’t poke _leaflet_id)
          const old = document.getElementById('map');
          const fresh = old.cloneNode(false);
          old.replaceWith(fresh);

          this.map = L.map('map', {
            center: [45.815, 15.981],
            zoom: 6,
            zoomControl: false,
            attributionControl: false,
            preferCanvas: true,
            zoomAnimation: false
          });

          this.map.createPane('areasPane'); this.map.getPane('areasPane').style.zIndex = 420;
          this.map.getPane('areasPane').style.pointerEvents = 'auto';
          this.map.createPane('routesPane'); this.map.getPane('routesPane').style.zIndex = 440;
          this.map.createPane('connectorsPane'); this.map.getPane('connectorsPane').style.zIndex = 460;
          this.map.createPane('markersPane'); this.map.getPane('markersPane').style.zIndex = 600;

          L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            maxZoom: 20, tileSize: 256, zoomOffset: 0, subdomains: 'abcd',
            attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
          }).addTo(this.map);

          L.control.zoom({position: 'bottomright'}).addTo(this.map);
          L.control.scale({maxWidth: 220, metric: true, imperial: true, position: 'bottomleft'}).addTo(this.map);

          // Groups
          this.drawnItems = L.featureGroup().addTo(this.map);
          this.groups.routes = L.featureGroup().addTo(this.map);
          this.groups.zones = L.featureGroup().addTo(this.map);
          this.groups.boxes = L.featureGroup().addTo(this.map);
          this.groups.connectors = L.featureGroup().addTo(this.map);

          // MarkerCluster group for markers
          this.groups.markers = L.markerClusterGroup({
            spiderfyOnMaxZoom: true,
            showCoverageOnHover: false,
            removeOutsideVisibleBounds: true,
            disableClusteringAtZoom: 16,
            pane: 'markersPane'
          }).addTo(this.map);

          // Events
          this.map.on('mousemove', (event) => {this.cursorLat = event.latlng.lat; this.cursorLng = event.latlng.lng; this.handleMapMouseMove(event);});
          this.map.on('zoomend', () => {this.zoomLevel = this.map.getZoom(); this.refreshOverlayMeta();});
          this.map.on('moveend', () => {this.refreshOverlayMeta();});
          this.map.on('click', (event) => {this.handleMapClick(event);});
          this.map.on('dblclick', (event) => {this.handleMapDoubleClick(event);});
          this.map.on('contextmenu', (event) => {
            if (this.isDrawingMode(this.mode) && this.currentDraft.points.length > 0) {
              event?.originalEvent?.preventDefault?.();
              this.undoLastDraftPoint();
            }
          });
          this.map.on('dragstart', () => {
            if (!this.map) return;
            const container = this.map.getContainer?.();
            if (container && this.mode === 'pan') container.style.cursor = 'grabbing';
          });
          this.map.on('dragend', () => {this.updateMapCursor();});
          this.map.on('zoomstart', () => {this.map?.closePopup?.();});
          this.zoomLevel = this.map.getZoom();
          this.map.doubleClickZoom?.enable?.();

          setTimeout(() => {this.map?.invalidateSize?.(true);}, 80);

          if (!this.resizeHandler) {
            this.resizeHandler = () => {this.map?.invalidateSize?.(false);};
            window.addEventListener('resize', this.resizeHandler, {passive: true});
          }
          this.resetDraft(this.isDrawingMode(this.mode) ? this.mode : null);
          this.updateMapCursor();
        },

        bindKeyControls() {
          if (this.keydownHandler) return;
          this.keydownHandler = this.handleKeydown.bind(this);
          window.addEventListener('keydown', this.keydownHandler);
        },

        updateStatus(message, detail = '') {this.status.message = message; this.status.detail = detail;},

        updateNetworkStatus() {
          if (typeof window === 'undefined' || !window.location) {
            this.networkStatusLabel = 'UNKNOWN';
            this.networkStatusClass = 'text-slate-300';
            return;
          }
          const protocol = (window.location.protocol || '').replace(':', '').toUpperCase();
          this.networkStatusLabel = protocol || 'UNKNOWN';
          this.networkStatusClass = protocol === 'HTTPS' ? 'text-mission-success' : 'text-mission-warning';
        },

        toggleLayers() {
          this.layersOpen = !this.layersOpen;
          if (this.layersOpen) this.applySheetHeight(0.5);
          else {
            this.finishSheetResize();
            this.cancelPendingDelete();
            this.closeFilterDialog();
            this.closeToolDropdown();
          }
        },

        handleLayerFocus(id) {
          this.closeOverlapChooser();
          setTimeout(() => {this.focusLayer(id);}, 10);
        },

        handleLayerEdit(id) {
          if (this.lockedOverlayId && this.lockedOverlayId !== id) return;
          this.closeOverlapChooser();
          setTimeout(() => {this.toggleEdit(id);}, 10);
        },

        handleLayerDelete(id) {
          if (this.lockedOverlayId && this.lockedOverlayId !== id) return;
          if (this.editingLayerId) return;
          this.closeOverlapChooser();
          setTimeout(() => {this.confirmRemoveOverlay(id);}, 10);
        },

        editingOverlaySummary() {
          if (!this.editingLayerId) return '';
          const overlay = this.overlays.find((item) => item.id === this.editingLayerId);
          if (!overlay) return '';
          const type = (overlay.type || '').toUpperCase();
          const label = overlay.label || 'Overlay';
          return `Editing ${label}${type ? ` (${type})` : ''}`;
        },

        updateMapCursor() {
          if (!this.map || typeof this.map.getContainer !== 'function') return;
          const container = this.map.getContainer();
          if (!container) return;
          if (this.mode === 'pan') container.style.cursor = 'grab';
          else container.style.cursor = 'crosshair';
        },

        applyMarkerIcon(layer, {label, color, symbol}) {
          if (!layer) return;
          const html = '<div class="dot-marker"></div>';
          layer.setIcon(L.divIcon({
            className: '',
            html,
            iconSize: [18, 18],
            iconAnchor: [9, 9],
            popupAnchor: [0, -12]
          }));
          this.setLayerPointerEvents(layer, this.mode === 'pan');
          const overlayId = layer.__overlayId ?? L.Util.stamp(layer);
          this.tagLayerElement(layer, overlayId);
        },

        setLayerPointerEvents(layer, interactive) {
          if (!layer) return;
          const value = interactive ? '' : 'none';
          if (layer._icon) layer._icon.style.pointerEvents = value;
          if (layer._shadow) layer._shadow.style.pointerEvents = value;
          if (layer._path) layer._path.style.pointerEvents = interactive ? 'auto' : 'none';
          if (layer._container && !layer._path) layer._container.style.pointerEvents = value;
        },

        updateAllLayerInteractivity() {
          const interactive = this.mode === 'pan';
          Object.values(this.layerIndex).forEach((layer) => {
            this.setLayerPointerEvents(layer, interactive);
          });
        },

        tagLayerElement(layer, id) {
          const assign = () => {
            const el = this.layerElement(layer);
            if (!el) return false;
            el.dataset.overlayId = String(id);
            return true;
          };
          if (!assign()) {
            setTimeout(assign, 20);
          }
        },

        layerElement(layer) {
          if (!layer) return null;
          if (typeof layer.getElement === 'function') return layer.getElement();
          if (layer._icon) return layer._icon;
          if (layer._path) return layer._path;
          return null;
        },

        handleOverlayClick(event, id, layer) {
          this.selectedLayerId = id;
          const overlay = this.overlays.find((item) => item.id === id);
          const original = event?.originalEvent;
          if (this.mode !== 'pan') {
            if (original && event?.latlng && this.map) {
              original.preventDefault?.();
              original.stopPropagation?.();
              this.map.fire('click', {latlng: event.latlng, originalEvent: original});
            }
            return;
          }

          const overlaps = this.collectOverlaysAtEvent(original, id, event?.latlng);
          if (overlaps.length <= 1) {
            this.closeOverlapChooser();
            this.openOverlayPopup(overlay, layer, event?.latlng);
            if (original) original._atakHandled = true;
            return;
          }

          original?.preventDefault?.();
          original?.stopPropagation?.();
          if (original) original._atakHandled = true;
          this.map?.closePopup?.();
          layer?.closePopup?.();
          overlaps.forEach((overlay) => {
            const targetLayer = this.layerIndex[overlay.id];
            targetLayer?.closePopup?.();
          });
          this.showOverlapChooser(original, overlaps);
        },

        collectOverlaysAtEvent(originalEvent, fallbackId, latlng = null) {
          const overlays = [];
          const seen = new Set();
          if (originalEvent && typeof document.elementsFromPoint === 'function') {
            const nodes = document.elementsFromPoint(originalEvent.clientX, originalEvent.clientY) || [];
            nodes.forEach((node) => {
              const id = node?.dataset?.overlayId;
              if (!id) return;
              const match = this.overlays.find((o) => String(o.id) === id);
              if (match && !seen.has(match.id)) {
                overlays.push(match);
                seen.add(match.id);
              }
            });
          }

          if (latlng) {
            this.overlays.forEach((overlay) => {
              if (seen.has(overlay.id)) return;
              if (this.overlayHitTest(overlay, latlng)) {
                overlays.push(overlay);
                seen.add(overlay.id);
              }
            });
          }

          if ((!overlays.length || !seen.has(fallbackId)) && fallbackId) {
            const match = this.overlays.find((o) => o.id === fallbackId);
            if (match && !seen.has(match.id)) {
              overlays.push(match);
              seen.add(match.id);
            }
          }
          return overlays;
        },

        overlayHitTest(overlay, latlng) {
          if (!overlay || !latlng || !this.map) return false;
          const layer = this.layerIndex[overlay.id];
          if (!layer) return false;
          const ll = L.latLng(latlng);
          if (!this.map.latLngToLayerPoint) return false;
          if (overlay.type === 'marker') {
            if (typeof layer.getLatLng !== 'function') return false;
            const markerPoint = this.map.latLngToLayerPoint(layer.getLatLng());
            const testPoint = this.map.latLngToLayerPoint(ll);
            return markerPoint.distanceTo(testPoint) <= 16;
          }
          const point = this.map.latLngToLayerPoint(ll);
          if (typeof layer._containsPoint === 'function' && layer._containsPoint(point)) return true;
          if (typeof layer.getBounds === 'function' && layer.getBounds().contains(ll)) return true;
          if (typeof layer.getLatLngs === 'function') {
            const latlngs = layer.getLatLngs();
            if (Array.isArray(latlngs) && latlngs.length) {
              const rings = this.normalizePolygonRings(latlngs);
              if (rings.some((ring) => this.pointInRing(ll, ring))) return true;
            }
          }
          return false;
        },

        showOverlapChooser(originalEvent, overlays) {
          const container = this.map?.getContainer?.();
          if (!container) return;
          this.closeOverlapChooser();
          const rect = container.getBoundingClientRect();
          const width = Math.min(220, rect.width - 16);
          const x = originalEvent.clientX - rect.left;
          const y = originalEvent.clientY - rect.top + 10;
          let left = x - width / 2;
          left = Math.max(8, Math.min(left, rect.width - width - 8));
          let top = y;
          const maxTop = rect.height - 160;
          if (top > maxTop) top = maxTop;
          const sorted = overlays.slice().sort((a, b) => (a.label || '').localeCompare(b.label || ''));
          this.overlapChooser = {
            open: true,
            items: sorted.map((item) => ({id: item.id, label: item.label, type: item.type})),
            style: {left: `${left}px`, top: `${top}px`, minWidth: `${width}px`}
          };
          this.closeFilterDialog();
          this.closeToolDropdown();
        },

        closeOverlapChooser() {
          if (this.overlapChooser.open) this.overlapChooser.open = false;
          if (this.overlapChooser.items.length) this.overlapChooser.items = [];
        },

        selectOverlapOverlay(id) {
          this.closeOverlapChooser();
          const layer = this.layerIndex[id];
          const overlay = this.overlays.find((item) => item.id === id);
          if (layer) {
            this.selectedLayerId = id;
            this.openOverlayPopup(overlay, layer);
          }
        },

        filteredOverlays() {
          if (this.layerFilter === 'all') return this.overlays;
          const typeMap = {marker: 'marker', route: 'polyline', zone: 'polygon', box: 'rectangle'};
          const target = typeMap[this.layerFilter];
          if (!target) return this.overlays;
          return this.overlays.filter((overlay) => overlay.type === target);
        },

        currentFilterOption() {
          return this.filterOptions.find((opt) => opt.value === this.layerFilter) || this.filterOptions[0];
        },

        toggleFilterDialog(event) {
          if (this.filterDialogOpen) this.closeFilterDialog();
          else {
            this.closeToolDropdown();
            this.closeOverlapChooser();
            this.openFilterDialog(event);
          }
        },

        openFilterDialog(event) {
          const dialog = this.$refs?.filterDialog;
          const button = this.$refs?.filterButton;
          if (!dialog) return;
          if (button) {
            const host = button.offsetParent;
            const hostWidth = host ? host.clientWidth : window.innerWidth;
            const minWidth = Math.min(Math.max(button.offsetWidth, 160), Math.max(hostWidth - 8, 120));
            const top = button.offsetTop + button.offsetHeight + 6;
            let left = button.offsetLeft + (button.offsetWidth / 2) - (minWidth / 2);
            const limit = hostWidth - minWidth - 4;
            if (left < 0) left = 0;
            if (left > limit) left = Math.max(0, limit);
            dialog.style.top = `${top}px`;
            dialog.style.left = `${left}px`;
            dialog.style.right = 'auto';
            dialog.style.minWidth = `${minWidth}px`;
          }
          if (!this.filterOutsideHandler) {
            this.filterOutsideHandler = (evt) => {
              const dlg = this.$refs?.filterDialog;
              const btn = this.$refs?.filterButton;
              if (dlg && dlg.contains(evt.target)) return;
              if (btn && btn.contains(evt.target)) return;
              this.closeFilterDialog();
            };
          }
          window.addEventListener('pointerdown', this.filterOutsideHandler, true);
          this.filterDialogOpen = true;
        },

        closeFilterDialog() {
          if (this.filterOutsideHandler) {
            window.removeEventListener('pointerdown', this.filterOutsideHandler, true);
            this.filterOutsideHandler = null;
          }
          this.filterDialogOpen = false;
        },

        setLayerFilter(value) {
          this.layerFilter = value;
          this.closeFilterDialog();
        },

        currentToolOption() {
          return this.toolOptions.find((opt) => opt.value === this.selectedTool) || {value: null, label: 'Tools', icon: 'fa-screwdriver-wrench'};
        },

        toggleToolDropdown(event) {
          if (this.toolDropdownOpen) this.closeToolDropdown();
          else {
            this.closeFilterDialog();
            this.openToolDropdown(event);
          }
        },

        openToolDropdown(event) {
          const dialog = this.$refs?.toolDialog;
          const button = this.$refs?.toolButton;
          if (!dialog || !button) return;
          const host = button.offsetParent;
          const hostWidth = host ? host.clientWidth : window.innerWidth;
          const minWidth = Math.min(Math.max(button.offsetWidth, 180), Math.max(hostWidth - 8, 140));
          const top = button.offsetTop + button.offsetHeight + 6;
          let left = button.offsetLeft + (button.offsetWidth / 2) - (minWidth / 2);
          const limit = hostWidth - minWidth - 4;
          if (left < 0) left = 0;
          if (left > limit) left = Math.max(0, limit);
          dialog.style.top = `${top}px`;
          dialog.style.left = `${left}px`;
          dialog.style.right = 'auto';
          dialog.style.minWidth = `${minWidth}px`;
          if (!this.toolOutsideHandler) {
            this.toolOutsideHandler = (evt) => {
              const dlg = this.$refs?.toolDialog;
              const btn = this.$refs?.toolButton;
              if (dlg && dlg.contains(evt.target)) return;
              if (btn && btn.contains(evt.target)) return;
              this.closeToolDropdown();
            };
          }
          window.addEventListener('pointerdown', this.toolOutsideHandler, true);
          this.toolDropdownOpen = true;
        },

        closeToolDropdown() {
          if (this.toolOutsideHandler) {
            window.removeEventListener('pointerdown', this.toolOutsideHandler, true);
            this.toolOutsideHandler = null;
          }
          this.toolDropdownOpen = false;
        },

        setTool(mode) {
          this.setMode(mode);
          this.closeToolDropdown();
        },

        finishActiveTool() {
          if (this.isDrawingMode(this.mode)) {
            if (this.currentDraft.points.length === 0) {this.setMode('pan'); return;}
            const success = this.finalizeDraft();
            if (!success) return;
          }
          this.setMode('pan');
          this.closeToolDropdown();
          this.closeOverlapChooser();
        },

        cancelActiveTool() {
          if (this.currentDraft.type) this.resetDraft(null, true);
          this.setMode('pan');
          this.closeToolDropdown();
          this.closeOverlapChooser();
        },

        applySheetHeight(height) {
          const clamped = Math.min(0.75, Math.max(0.25, Number.isFinite(height) ? height : 0.5));
          this.sheetHeight = clamped;
          const dialog = this.$refs?.layersDialog;
          if (dialog) {
            dialog.style.setProperty('--sheet-height', clamped);
            dialog.style.height = `calc(${clamped} * 100vh)`;
          }
        },

        beginSheetResize(event) {
          if (!this.layersOpen) return;
          this.sheetDragActive = true;
          this.sheetDragStartY = event?.clientY ?? 0;
          this.sheetStartHeight = this.sheetHeight || 0.5;
          if (!this.sheetResizeHandlers.move) {
            this.sheetResizeHandlers.move = (e) => this.resizeSheet(e);
            this.sheetResizeHandlers.up = (e) => this.finishSheetResize(e);
          }
          window.addEventListener('pointermove', this.sheetResizeHandlers.move, {passive: false});
          window.addEventListener('pointerup', this.sheetResizeHandlers.up, {passive: true});
          window.addEventListener('pointercancel', this.sheetResizeHandlers.up, {passive: true});
        },

        resizeSheet(event) {
          if (!this.sheetDragActive) return;
          const viewportHeight = window.innerHeight || document.documentElement?.clientHeight || 1;
          const delta = (event?.clientY ?? 0) - this.sheetDragStartY;
          const ratioDelta = viewportHeight ? -delta / viewportHeight : 0;
          const nextHeight = this.sheetStartHeight + ratioDelta;
          this.applySheetHeight(nextHeight);
          event?.preventDefault?.();
        },

        finishSheetResize() {
          if (!this.sheetDragActive) return;
          this.sheetDragActive = false;
          window.removeEventListener('pointermove', this.sheetResizeHandlers.move);
          window.removeEventListener('pointerup', this.sheetResizeHandlers.up);
          window.removeEventListener('pointercancel', this.sheetResizeHandlers.up);
        },

        isDrawingMode(mode) {return ['polyline', 'polygon', 'rectangle'].includes(mode);},

        resetDraft(nextType = null, notify = false) {
          if (this.currentDraft.layer && this.map) this.map.removeLayer(this.currentDraft.layer);
          if (Array.isArray(this.currentDraft.markers)) this.currentDraft.markers.forEach((m) => m?.remove?.());
          this.currentDraft = {type: nextType, points: [], layer: null, markers: []};
          if (notify) this.updateStatus('Drawing cancelled', 'Draft overlay discarded.');
        },

        clearDraftMarkers() {
          if (!Array.isArray(this.currentDraft.markers)) return;
          this.currentDraft.markers.forEach((m) => m?.remove?.());
          this.currentDraft.markers = [];
        },

        addDraftMarker(latlng, index) {
          if (!this.map || !latlng) return;
          const marker = L.marker(latlng, {
            interactive: false, zIndexOffset: 600,
            icon: L.divIcon({className: '', html: `<div class="draft-marker">${index}</div>`, iconSize: [24, 24], iconAnchor: [12, 12]})
          });
          marker.addTo(this.map);
          this.currentDraft.markers.push(marker);
        },

        undoLastDraftPoint() {
          if (!this.isDrawingMode(this.mode) || this.currentDraft.points.length === 0) return;
          const marker = this.currentDraft.markers.pop(); if (marker) marker.remove();
          this.currentDraft.points.pop();
          if (this.currentDraft.points.length === 0) {this.resetDraft(null, true); this.setMode('pan');}
          else {this.updateDraftPreview(); this.updateStatus('Point removed', `Remaining vertices: ${this.currentDraft.points.length}`);}
        },

        setMode(mode) {
          const willDraw = this.isDrawingMode(mode);
          this.mode = mode;

          if (mode === 'pan') {
            this.selectedTool = null;
            this.closeToolDropdown();
            if (this.currentDraft.type) this.resetDraft(null, true);
          } else if (['marker', 'polyline', 'polygon', 'rectangle'].includes(mode)) {
            this.selectedTool = mode;
            if (!this.isDrawingMode(mode) && this.currentDraft.type) this.resetDraft(null, true);
          }

          if (mode !== 'pan' && this.editingLayerId) this.stopEditing();

          if (mode !== 'pan') {Object.values(this.layerIndex).forEach((l) => l?.closePopup?.()); this.map?.closePopup?.();}

          if (this.map && this.map.doubleClickZoom) {
            const control = this.map.doubleClickZoom;
            if (typeof control[willDraw ? 'disable' : 'enable'] === 'function') control[willDraw ? 'disable' : 'enable']();
          }

          if (mode === 'pan') {
            Object.values(this.layerIndex).forEach((layer) => {
              if (layer?.dragging && typeof layer.dragging.disable === 'function') layer.dragging.disable();
            });
          }

          if (mode !== 'pan') Object.values(this.layerIndex).forEach((l) => l?.closePopup?.());
          this.resetDraft(willDraw ? mode : null);
          this.closeOverlapChooser();

          if (mode === 'marker') this.updateStatus('Marker inject mode', 'Click map to deploy tactical marker.');
          else if (mode === 'polyline') this.updateStatus('Route plot mode', 'Left-click to add segments; double click or press Enter to finish.');
          else if (mode === 'polygon') this.updateStatus('Zone definition mode', 'Left-click to add vertices; double click or press Enter to close.');
          else if (mode === 'rectangle') this.updateStatus('Box perimeter mode', 'Click start corner, move pointer, click again to finalize.');
          else this.updateStatus('Navigation mode', 'Pan and zoom map freely.');
          this.updateMapCursor();
          this.updateAllLayerInteractivity();
        },

        handleMapClick(event) {
          if (!this.map) return;
          this.closeOverlapChooser();
          const latlng = L.latLng(event.latlng);
          if (this.mode === 'pan') {
            const nativeEvent = event?.originalEvent;
            if (nativeEvent && nativeEvent._atakHandled) return;
            const overlays = this.collectOverlaysAtEvent(nativeEvent, null, latlng);
            if (overlays.length) {
              if (nativeEvent) nativeEvent._atakHandled = true;
              if (overlays.length === 1) {
                const target = overlays[0];
                const layer = this.layerIndex[target.id];
                if (layer) {
                  this.selectedLayerId = target.id;
                  this.map?.closePopup?.();
                  this.openOverlayPopup(target, layer, latlng);
                }
              } else if (nativeEvent) {
                this.showOverlapChooser(nativeEvent, overlays);
              }
              return;
            }
          }
          if (this.mode === 'marker') {this.injectMarker(latlng); return;}
          if (!this.isDrawingMode(this.mode) || this.currentDraft.type !== this.mode) return;

          const draft = this.currentDraft;
          if (draft.type === 'rectangle') {
            if (draft.points.length === 0) {draft.points.push(latlng); this.addDraftMarker(latlng, draft.points.length); this.updateStatus('Perimeter anchor set', 'Move pointer, click again to close perimeter.'); return;}
            if (draft.points.length === 1) {draft.points.push(latlng); this.addDraftMarker(latlng, draft.points.length); this.updateDraftPreview(); this.finalizeDraft();}
            return;
          }

          draft.points.push(latlng);
          this.addDraftMarker(latlng, draft.points.length);
          this.updateDraftPreview();
          if (draft.points.length === 1) this.updateStatus('Geometry capture engaged', 'Continue plotting points. Double click or press Enter to lock.');
          else {
            const required = draft.type === 'polygon' ? 3 : 2;
            if (draft.points.length >= required) this.updateStatus('Geometry capture engaged', 'Double click or press Enter to finalize overlay.');
          }
        },

        handleMapDoubleClick(event) {
          if (!this.isDrawingMode(this.mode) || !this.currentDraft.type) return;
          const required = this.currentDraft.type === 'polygon' ? 3 : 2;
          if (this.currentDraft.points.length >= required) {
            if (event && event.originalEvent) {event.originalEvent.preventDefault(); event.originalEvent.stopPropagation?.();}
            this.finalizeDraft();
          }
        },

        handleMapMouseMove(event) {
          if (!this.isDrawingMode(this.mode) || !this.currentDraft.type) return;
          if (this.currentDraft.type === 'rectangle' && this.currentDraft.points.length === 1) this.updateDraftPreview(event.latlng);
          else if (['polyline', 'polygon'].includes(this.currentDraft.type) && this.currentDraft.points.length > 0) this.updateDraftPreview(event.latlng);
        },

        handleKeydown(event) {
          const target = event.target; const tag = target && target.tagName;
          if (target && (target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(tag))) return;
          const key = event.key.toLowerCase();
          if ((key === 'backspace' || key === 'delete') && this.isDrawingMode(this.mode) && this.currentDraft.points.length > 0) {event.preventDefault(); this.undoLastDraftPoint(); return;}
          if (key === 'escape') {
            if (this.filterDialogOpen) this.closeFilterDialog();
            if (this.toolDropdownOpen) this.closeToolDropdown();
            if (this.overlapChooser.open) this.closeOverlapChooser();
            if (this.layersOpen) {this.layersOpen = false;}
            if (this.currentDraft.type) this.resetDraft(this.isDrawingMode(this.mode) ? this.mode : null, true);
            if (this.editingLayerId) this.stopEditing();
            this.setMode('pan');
            event.preventDefault();
            return;
          }
          if (key === 'enter') {
            if (this.currentDraft.type) {
              event.preventDefault();
              const success = this.finalizeDraft();
              if (success) this.closeToolDropdown();
            }
            return;
          }
          if (key === 'm') this.setMode('marker');
          else if (key === 'r') this.setMode('polyline');
          else if (key === 'z') this.setMode('polygon');
          else if (key === 'b') this.setMode('rectangle');
          else if (key === 'p') this.setMode('pan');
        },

        updateDraftPreview(previewLatLng = null) {
          const draft = this.currentDraft;
          if (!draft.type) return;

          if (draft.type === 'rectangle') {
            if (draft.points.length === 0) return;
            const anchor = draft.points[0];
            const terminal = draft.points[1] || previewLatLng;
            if (!terminal) return;
            const bounds = L.latLngBounds(anchor, terminal);
            if (!draft.layer) {
              draft.layer = L.rectangle(bounds, {color: this.defaultColor('rectangle'), weight: 2, opacity: 0.95, fillOpacity: 0.2, dashArray: '6 4', pane: 'areasPane'}).addTo(this.map);
            } else draft.layer.setBounds(bounds);
            return;
          }

          const basePoints = draft.points.slice();
          if (previewLatLng) basePoints.push(previewLatLng);
          if (basePoints.length < 2) return;

          if (draft.type === 'polyline') {
            if (!draft.layer) draft.layer = L.polyline(basePoints, {color: this.defaultColor('polyline'), weight: 3, opacity: 0.9, dashArray: '6 4', pane: 'routesPane'}).addTo(this.map);
            else draft.layer.setLatLngs(basePoints);
            return;
          }

          if (draft.type === 'polygon') {
            if (basePoints.length < 3) {
              if (draft.layer && !(draft.layer instanceof L.Polyline) && draft.layer.remove) {this.map.removeLayer(draft.layer); draft.layer = null;}
              if (!draft.layer) draft.layer = L.polyline(basePoints, {color: this.defaultColor('polygon'), weight: 3, opacity: 0.92, dashArray: '6 4', pane: 'areasPane'}).addTo(this.map);
              else draft.layer.setLatLngs(basePoints);
              return;
            }
            if (draft.layer && !(draft.layer instanceof L.Polygon)) {this.map.removeLayer(draft.layer); draft.layer = null;}
            if (!draft.layer) draft.layer = L.polygon(basePoints, {color: this.defaultColor('polygon'), weight: 3, opacity: 0.92, fillOpacity: 0.24, dashArray: '6 4', pane: 'areasPane'}).addTo(this.map);
            else draft.layer.setLatLngs(basePoints);
          }
        },

        finalizeDraft() {
          const draft = this.currentDraft;
          if (!draft.type) return false;

          const points = draft.points.slice();
          if (draft.type === 'polyline' && points.length < 2) {this.updateStatus('Route incomplete', 'Minimum two points required to generate a route.'); return false;}
          if (draft.type === 'polygon' && points.length < 3) {this.updateStatus('Zone incomplete', 'Minimum three vertices required to close zone.'); return false;}
          if (draft.type === 'rectangle' && points.length < 2) {this.updateStatus('Perimeter incomplete', 'Select two opposing corners to define perimeter.'); return false;}

          if (draft.layer && this.map) this.map.removeLayer(draft.layer);

          if (draft.type === 'polyline') {
            const layer = L.polyline(points, {color: this.defaultColor('polyline'), weight: 3, opacity: 0.9, pane: 'routesPane'});
            layer.addTo(this.drawnItems); layer.addTo(this.groups.routes);
            this.registerLayer(layer, 'polyline'); layer.bringToFront?.();
            const metrics = this.calculatePolyline(layer);
            this.updateStatus('Route plotted', `Length ${this.formatLength(metrics.lengthM)}`);
          } else if (draft.type === 'polygon') {
            const layer = L.polygon(points, {color: this.defaultColor('polygon'), weight: 3, opacity: 0.95, fillOpacity: 0.28, pane: 'areasPane'});
            layer.addTo(this.drawnItems); layer.addTo(this.groups.zones);
            this.registerLayer(layer, 'polygon'); layer.bringToFront?.();
            const metrics = this.calculatePolygon(layer);
            this.updateStatus('Zone defined', `Area ${this.formatArea(metrics.areaSqM)}`);
          } else if (draft.type === 'rectangle') {
            const bounds = L.latLngBounds(points[0], points[1]);
            const layer = L.rectangle(bounds, {color: this.defaultColor('rectangle'), weight: 3, opacity: 0.95, fillOpacity: 0.24, pane: 'areasPane'});
            layer.addTo(this.drawnItems); layer.addTo(this.groups.boxes);
            this.registerLayer(layer, 'rectangle'); layer.bringToFront?.();
            const metrics = this.calculateRectangle(layer);
            this.updateStatus('Box perimeter established', `Area ${this.formatArea(metrics.areaSqM)}`);
          }

          this.clearDraftMarkers();
          this.resetDraft(this.isDrawingMode(this.mode) ? this.mode : null);
          return true;
        },

        injectMarker(latlng, options = {}) {
          const label = options.label || this.markerForm.label?.trim() || this.autoLabel('marker');
          const color = options.color || this.markerForm.color || this.defaultColor('marker');
          const symbol = options.symbol || this.markerForm.symbol || '▲';
          const persistLabel = options.persist ?? this.markerForm.persist;

          const marker = L.marker(latlng, {pane: 'markersPane'});
          this.applyMarkerIcon(marker, {label, color, symbol});

          // Popup & add to layers
          marker.bindPopup(this.buildPopup({label, type: 'marker', latlng}));
          marker.addTo(this.drawnItems);
          this.groups.markers.addLayer(marker);   // <-- cluster group

          // Not draggable by default
          marker.dragging?.disable?.();

          marker.off('dragend._marker');
          marker.on('dragend._marker', () => {
            const pos = marker.getLatLng();
            this.refreshOverlayMeta();
            this.updateStatus('Marker repositioned', `${label} at ${this.formatLat(pos.lat)}, ${this.formatLng(pos.lng)}`);
          });

          this.registerLayer(marker, 'marker', {label, color, symbol, latlng});

          if (!persistLabel && !options.skipReset) this.markerForm.label = '';
          this.updateStatus('Marker deployed', `${label} at ${this.formatLat(latlng.lat)}, ${this.formatLng(latlng.lng)}`);
        },

        registerLayer(layer, type, meta = {}) {
          const id = L.Util.stamp(layer);
          this.layerIndex[id] = layer;
          layer.__overlayId = id;

          let overlayMeta = {};
          if (type === 'marker') {
            const position = meta.latlng || layer.getLatLng();
            overlayMeta = {lat: position.lat, lng: position.lng, color: meta.color, symbol: meta.symbol, label: meta.label};
          } else if (type === 'polyline' || type === 'connector') {
            overlayMeta = this.calculatePolyline(layer);
          } else if (type === 'polygon') {
            overlayMeta = this.calculatePolygon(layer);
          } else if (type === 'rectangle') {
            overlayMeta = this.calculateRectangle(layer);
          }

          const label = meta.label || this.autoLabel(type);

          if (type === 'connector') {
            const startPoint = meta.startLatLng || this.getOverlayLatLng(meta.startId);
            const endPoint = meta.endLatLng || this.getOverlayLatLng(meta.endId);
            overlayMeta = {
              ...overlayMeta,
              startId: meta.startId,
              endId: meta.endId,
              style: meta.style || 'solid',
              direction: meta.direction || 'none',
              color: meta.color || this.defaultColor('polyline'),
              startLat: startPoint?.lat, startLng: startPoint?.lng,
              endLat: endPoint?.lat, endLng: endPoint?.lng
            };
            this.applyConnectorStyling(layer, overlayMeta);
            this.groups.connectors.addLayer(layer);
          } else if (type !== 'marker' && layer.setStyle) {
            layer.setStyle({color: overlayMeta.color || this.defaultColor(type)});
          }

          layer.bindPopup(this.buildPopup({label, type, latlng: layer.getLatLng?.(), meta: overlayMeta}), this.popupOptions(type));

          layer.off('click', layer.openPopup, layer);
          layer.on('click', (event) => { this.handleOverlayClick(event, id, layer); });
          layer.on('add', () => { this.tagLayerElement(layer, id); });
          this.tagLayerElement(layer, id);

          this.overlays.push({id, type, label, created: new Date().toISOString(), ...overlayMeta});
          if (type === 'connector') this.updateConnectorArrows(id);
          this.selectedLayerId = id;
          this.setLayerPointerEvents(layer, this.mode === 'pan');
        },

        popupOptions(type) {
          const padding = L.point(32, 24);
          if (type === 'marker') return {autoPanPadding: padding};
          return {autoPanPadding: padding};
        },

        openOverlayPopup(overlay, layer, latlng = null) {
          if (!layer) return;
          const type = overlay?.type;
          if (type === 'marker') {
            layer.openPopup?.();
            return;
          }
          if (latlng) {
            layer.openPopup?.(latlng);
            return;
          }
          layer.openPopup?.();
        },

        connectorDashArray(style) {if (style === 'dashed') return '10 6'; if (style === 'dotted') return '2 10'; return null;},
        applyConnectorStyling(layer, meta) {
          if (!layer?.setStyle) return;
          layer.setStyle({color: meta.color || this.defaultColor('polyline'), weight: 3, opacity: 0.95, dashArray: this.connectorDashArray(meta.style), lineCap: 'round', lineJoin: 'round'});
          layer.bringToFront?.();
        },

        updateConnectorArrows(id) {
          const overlay = this.overlays.find((i) => i.id === id);
          const layer = this.layerIndex[id];
          if (!overlay || !layer || overlay.type !== 'connector' || !this.map) return;
          this.removeConnectorArtifacts(id);
          const start = overlay.startLat !== undefined ? L.latLng(overlay.startLat, overlay.startLng) : this.getOverlayLatLng(overlay.startId);
          const end = overlay.endLat !== undefined ? L.latLng(overlay.endLat, overlay.endLng) : this.getOverlayLatLng(overlay.endId);
          if (!start || !end) return;
          this.connectorArtifacts[id] = this.connectorArtifacts[id] || {startMarker: null, endMarker: null};
          const startPoint = this.map.latLngToLayerPoint(start);
          const endPoint = this.map.latLngToLayerPoint(end);
          const angle = Math.atan2(endPoint.y - startPoint.y, endPoint.x - startPoint.x);
          const color = overlay.color || this.defaultColor('polyline');
          if (overlay.direction === 'start' || overlay.direction === 'both') this.connectorArtifacts[id].startMarker = this.createArrowMarker(start, angle + Math.PI, color);
          if (overlay.direction === 'end' || overlay.direction === 'both') this.connectorArtifacts[id].endMarker = this.createArrowMarker(end, angle, color);
        },

        removeConnectorArtifacts(id) {
          const artifacts = this.connectorArtifacts[id];
          if (!artifacts) return;
          artifacts.startMarker?.remove?.();
          artifacts.endMarker?.remove?.();
          delete this.connectorArtifacts[id];
        },

        createArrowMarker(latlng, angle, color) {
          if (!this.map) return null;
          const marker = L.marker(latlng, {
            interactive: false, zIndexOffset: 700,
            icon: L.divIcon({className: '', html: `<div class="connector-arrow" style="border-bottom-color:${color}; transform: rotate(${angle}rad);"></div>`, iconSize: [24, 24], iconAnchor: [12, 14]})
          });
          marker.addTo(this.map);
          return marker;
        },

        connectableOverlays() {return this.overlays.filter((o) => o.type !== 'connector').sort((a, b) => a.label.localeCompare(b.label));},
        findOverlayLabel(id) {if (!id) return '—'; const o = this.overlays.find((i) => i.id === id); return o?.label || `ID ${id}`;},

        getOverlayLatLng(id) {
          const layer = this.layerIndex[id];
          const overlay = this.overlays.find((i) => i.id === id);
          if (!layer || !overlay) return null;
          if (overlay.type === 'marker') return layer.getLatLng ? layer.getLatLng() : L.latLng(overlay.lat, overlay.lng);
          if (overlay.type === 'polyline') {
            const latlngs = layer.getLatLngs?.(); if (!latlngs || latlngs.length === 0) return null;
            const flattened = Array.isArray(latlngs[0]) ? latlngs[0] : latlngs; if (!flattened || flattened.length === 0) return null;
            if (this.connectorForm.anchorMode === 'polyline-end') return flattened[flattened.length - 1];
            if (this.connectorForm.anchorMode === 'polyline-start') return flattened[0];
            return this.polylineMidpoint({getLatLngs: () => flattened});
          }
          if (overlay.type === 'connector') {
            const latlngs = layer.getLatLngs?.(); if (!latlngs || latlngs.length === 0) return null;
            return Array.isArray(latlngs[0]) ? this.polylineMidpoint({getLatLngs: () => latlngs[0]}) : this.polylineMidpoint(layer);
          }
          if (overlay.type === 'polygon' || overlay.type === 'rectangle') return layer.getBounds ? layer.getBounds().getCenter() : L.latLng(overlay.lat, overlay.lng);
          return null;
        },

        polylineMidpoint(layer) {
          if (!layer?.getLatLngs) return null;
          const latlngs = layer.getLatLngs(); if (!Array.isArray(latlngs) || latlngs.length === 0) return null;
          if (Array.isArray(latlngs[0])) return this.polylineMidpoint({getLatLngs: () => latlngs[0]});
          let total = 0; const segments = [];
          for (let i = 0; i < latlngs.length - 1; i += 1) {const seg = latlngs[i].distanceTo(latlngs[i + 1]); total += seg; segments.push({length: seg, start: latlngs[i], end: latlngs[i + 1]});}
          let target = total / 2;
          for (const s of segments) {
            if (target <= s.length) {const ratio = s.length === 0 ? 0 : target / s.length; const lat = s.start.lat + (s.end.lat - s.start.lat) * ratio; const lng = s.start.lng + (s.end.lng - s.start.lng) * ratio; return L.latLng(lat, lng);}
            target -= s.length;
          }
          return latlngs[Math.floor(latlngs.length / 2)] || null;
        },

        createConnector() {this.updateStatus('Connector builder on hold', 'Linking tool is temporarily disabled.'); return;},

        toggleEdit(id) {if (this.editingLayerId === id) this.stopEditing(); else this.startEditing(id);},

        completeEditing() {
          if (!this.editingLayerId) return;
          const overlay = this.overlays.find((i) => i.id === this.editingLayerId);
          const layer = overlay ? this.layerIndex[overlay.id] : null;
          this.stopEditing({skipStatus: true});
          if (overlay?.type === 'marker' && layer) {
            layer.dragging?.disable?.();
            layer.off('dragend._editMarker');
            this.applyMarkerIcon(layer, {label: overlay.label, color: overlay.color || this.defaultColor('marker'), symbol: overlay.symbol || '▲'});
          }
          const label = overlay?.label || 'Overlay';
          this.updateStatus('Edits saved', `${label} updated.`);
        },

        cancelEditing() {
          if (!this.editingLayerId) return;
          this.restoreEditingSnapshot();
          const overlay = this.overlays.find((i) => i.id === this.editingLayerId);
          const layer = overlay ? this.layerIndex[overlay.id] : null;
          this.stopEditing({skipStatus: true});
          if (overlay?.type === 'marker' && layer) {
            layer.dragging?.disable?.();
            layer.off('dragend._editMarker');
            this.applyMarkerIcon(layer, {label: overlay.label, color: overlay.color || this.defaultColor('marker'), symbol: overlay.symbol || '▲'});
          }
          const label = overlay?.label || 'Overlay';
          this.updateStatus('Edit cancelled', `${label} reverted.`);
        },

        startEditing(id) {
          const overlay = this.overlays.find((i) => i.id === id);
          const layer = this.layerIndex[id];
          if (!overlay || !layer) return;
          if (overlay.type === 'connector') {this.updateStatus('Edit unavailable', 'Connector editing is disabled.'); return;}
          if (this.mode !== 'pan') this.setMode('pan');
          this.stopEditing({skipStatus: true});
          this.cancelPendingDelete();
          this.closeFilterDialog();
          this.closeOverlapChooser();
          if (!this.layersOpen) this.layersOpen = true;
          this.lockedOverlayId = id;
          if (overlay.type === 'marker') {
            this.editingSnapshot = this.captureEditingSnapshot(overlay, layer);
            this.enableMarkerEditing(layer, overlay);
            return;
          }
          const points = this.extractEditablePoints(layer, overlay.type);
          this.editingSnapshot = this.captureEditingSnapshot(overlay, layer, points);
          this.editingLayerId = id; this.selectedLayerId = id;
          this.editingData = {id, type: overlay.type, layer, points, bounds: overlay.type === 'rectangle' ? layer.getBounds() : null};
          layer.bringToFront?.(); this.buildEditingMarkers();
          if (layer.setStyle) layer.setStyle({dashArray: '4 4', weight: overlay.type === 'polyline' ? 4 : 3});
          this.updateStatus('Edit mode engaged', `Adjust vertices for ${overlay.label}. Drag handles or right-click to remove.`);
        },

        enableMarkerEditing(layer, overlay) {
          if (!this.editingSnapshot) this.editingSnapshot = this.captureEditingSnapshot(overlay, layer);
          this.editingLayerId = overlay.id; this.selectedLayerId = overlay.id;
          this.editingData = {id: overlay.id, type: 'marker', layer, points: [layer.getLatLng?.() || L.latLng(overlay.lat, overlay.lng)]};
          layer.dragging?.enable?.(); layer.bringToFront?.();
          this.applyMarkerIcon(layer, {label: overlay.label, color: '#facc15', symbol: overlay.symbol || '▲'});
          layer.off('dragend._editMarker');
          layer.on('dragend._editMarker', () => {
            const pos = layer.getLatLng(); this.refreshOverlayMeta();
            this.updateStatus('Marker repositioned', `${overlay.label} at ${this.formatLat(pos.lat)}, ${this.formatLng(pos.lng)}`);
          });
          this.updateStatus('Marker edit mode', `Drag ${overlay.label} to reposition. Tap Done when complete.`);
        },

        captureEditingSnapshot(overlay, layer, points = null) {
          if (!overlay || !layer) return null;
          if (overlay.type === 'marker') {
            const pos = layer.getLatLng?.() || L.latLng(overlay.lat, overlay.lng);
            return pos ? {id: overlay.id, type: 'marker', latlng: [pos.lat, pos.lng]} : null;
          }
          if (overlay.type === 'rectangle') {
            const bounds = layer.getBounds?.();
            if (!bounds) return null;
            const sw = bounds.getSouthWest();
            const ne = bounds.getNorthEast();
            return {id: overlay.id, type: 'rectangle', bounds: [[sw.lat, sw.lng], [ne.lat, ne.lng]]};
          }
          const sourcePoints = Array.isArray(points) ? points : this.extractEditablePoints(layer, overlay.type);
          if (!Array.isArray(sourcePoints)) return null;
          const serialized = sourcePoints.map((pt) => [pt.lat, pt.lng]);
          return {id: overlay.id, type: overlay.type, latlngs: serialized};
        },

        restoreEditingSnapshot() {
          const snapshot = this.editingSnapshot;
          if (!snapshot) return;
          const layer = this.layerIndex[snapshot.id];
          if (!layer) return;
          if (snapshot.type === 'marker' && Array.isArray(snapshot.latlng)) {
            const [lat, lng] = snapshot.latlng;
            if (lat !== undefined && lng !== undefined) layer.setLatLng?.(L.latLng(lat, lng));
          } else if (snapshot.type === 'rectangle' && Array.isArray(snapshot.bounds)) {
            const [southWest, northEast] = snapshot.bounds;
            if (southWest && northEast) {
              const sw = L.latLng(southWest[0], southWest[1]);
              const ne = L.latLng(northEast[0], northEast[1]);
              layer.setBounds?.(L.latLngBounds(sw, ne));
            }
          } else if (Array.isArray(snapshot.latlngs)) {
            const latlngs = snapshot.latlngs.map(([lat, lng]) => L.latLng(lat, lng));
            if (snapshot.type === 'polyline') layer.setLatLngs(latlngs);
            else if (snapshot.type === 'polygon') layer.setLatLngs([this.ensurePolygonPoints(latlngs)]);
          }
          this.refreshOverlayMeta();
        },

        stopEditing(options = {}) {
          const {skipStatus = false} = options;
          const wasEditing = this.editingLayerId !== null;
          const activeId = this.editingLayerId;
          if (this.editingMarkers.length) this.editingMarkers.forEach((m) => m?.remove?.());
          if (this.editingLayerId && this.layerIndex[this.editingLayerId]?.setStyle) {
            const layer = this.layerIndex[this.editingLayerId];
            const overlay = this.overlays.find((i) => i.id === this.editingLayerId);
            if (overlay?.type === 'polyline') layer.setStyle({dashArray: null, weight: 3, opacity: 0.9});
            else if (overlay?.type === 'polygon') layer.setStyle({dashArray: null, weight: 3, opacity: 0.95, fillOpacity: 0.28});
            else if (overlay?.type === 'rectangle') layer.setStyle({dashArray: null, weight: 3, opacity: 0.95, fillOpacity: 0.24});
            else if (overlay?.type === 'marker') {
              layer.off('dragend._editMarker');
              layer.dragging?.disable?.();
              this.applyMarkerIcon(layer, {label: overlay.label, color: overlay.color || this.defaultColor('marker'), symbol: overlay.symbol || '▲'});
            } else layer.setStyle?.({dashArray: null});
          }
          this.editingMarkers = []; this.editingLayerId = null; this.editingData = null; this.editingSnapshot = null;
          this.map?.closePopup?.(); this.refreshOverlayMeta();
          if (wasEditing && !skipStatus) this.updateStatus('Navigation mode', 'Pan and zoom map freely.');
          this.updateMapCursor();
          this.updateAllLayerInteractivity();
          if (!this.pendingDeleteId) this.lockedOverlayId = null;
          else if (this.pendingDeleteId !== activeId) this.lockedOverlayId = this.pendingDeleteId;
        },

        extractEditablePoints(layer, type) {
          if (type === 'rectangle') return this.rectangleCornersFromBounds(layer.getBounds());
          if (type === 'polyline') {
            const latlngs = layer.getLatLngs();
            const base = Array.isArray(latlngs[0]) ? latlngs[0] : latlngs;
            return base.map((ll) => L.latLng(ll));
          }
          const latlngs = layer.getLatLngs()[0] || [];
          const pts = latlngs.map((ll) => L.latLng(ll));
          if (pts.length > 1 && pts[pts.length - 1].equals(pts[0])) pts.pop();
          return pts;
        },

        buildEditingMarkers() {
          if (!this.map) return;
          this.editingMarkers.forEach((m) => m?.remove?.());
          this.editingMarkers = [];
          if (!this.editingData) return;

          if (this.editingData.type === 'marker') {
            const markerLayer = this.editingData.layer;
            markerLayer.dragging?.enable?.();
            markerLayer.off('dragend._editMarker');
            markerLayer.on('dragend._editMarker', () => {
              this.refreshOverlayMeta();
              const pos = markerLayer.getLatLng();
              this.updateStatus('Marker repositioned', `${this.findOverlayLabel(this.editingData.id)} at ${this.formatLat(pos.lat)}, ${this.formatLng(pos.lng)}`);
            });
            return;
          }

          const {type, points} = this.editingData;
          points.forEach((latlng, index) => {
            const marker = L.marker(latlng, {
              draggable: true,
              icon: L.divIcon({className: '', html: `<div class="draft-marker">${index + 1}</div>`, iconSize: [26, 26], iconAnchor: [13, 13]})
            });

            marker.on('drag', (event) => {
              const newLatLng = event.target.getLatLng();
              this.updateEditingPoint(index, newLatLng, {silent: type === 'rectangle'});
              if (type === 'rectangle') this.updateRectangleMarkerPositions();
            });

            marker.on('dragend', () => {this.refreshOverlayMeta(); this.buildEditingMarkers();});
            marker.on('click', (e) => e.originalEvent?.stopPropagation?.());
            marker.on('contextmenu', (event) => {
              event.originalEvent?.preventDefault?.();
              event.originalEvent?.stopPropagation?.();
              if (type === 'polyline' || type === 'polygon') this.removeEditingPoint(index);
            });

            marker.addTo(this.map);
            this.editingMarkers.push(marker);
          });
          if (type === 'rectangle') this.updateRectangleMarkerPositions();
        },

        updateEditingPoint(index, latlng, options = {}) {
          if (!this.editingData) return;
          const {type, layer} = this.editingData;
          if (type === 'marker') {layer.setLatLng(latlng); this.refreshOverlayMeta(); return;}
          if (type === 'rectangle') {
            const points = this.editingData.points;
            const opposite = points[(index + 2) % 4];
            const bounds = L.latLngBounds(latlng, opposite);
            layer.setBounds(bounds);
            this.editingData.points = this.rectangleCornersFromBounds(bounds);
            if (!options.silent) {this.updateRectangleMarkerPositions(); this.refreshOverlayMeta();}
            layer.bringToFront?.();
            return;
          }

          this.editingData.points[index] = latlng;
          if (type === 'polyline') layer.setLatLngs(this.editingData.points);
          else if (type === 'polygon') layer.setLatLngs([this.ensurePolygonPoints(this.editingData.points)]);
          layer.bringToFront?.();
          if (!options.silent) this.refreshOverlayMeta();
        },

        removeEditingPoint(index) {
          if (!this.editingData) return;
          const {type, layer} = this.editingData;
          const minPoints = type === 'polyline' ? 2 : 3;
          if (this.editingData.points.length <= minPoints) {this.updateStatus('Removal blocked', 'Maintain minimum vertices for this geometry.'); return;}
          this.editingData.points.splice(index, 1);
          if (type === 'polyline') layer.setLatLngs(this.editingData.points);
          else layer.setLatLngs([this.ensurePolygonPoints(this.editingData.points)]);
          layer.bringToFront?.();
          this.refreshOverlayMeta(); this.buildEditingMarkers();
          this.updateStatus('Vertex removed', `Remaining vertices: ${this.editingData.points.length}`);
        },

        ensurePolygonPoints(points) {if (!points.length) return []; const c = points.map((pt) => L.latLng(pt)); c.push(L.latLng(points[0])); return c;},
        rectangleCornersFromBounds(bounds) {return [L.latLng(bounds.getSouthWest()), L.latLng(bounds.getNorthWest()), L.latLng(bounds.getNorthEast()), L.latLng(bounds.getSouthEast())];},
        updateRectangleMarkerPositions() {
          if (!this.editingData || this.editingData.type !== 'rectangle') return;
          const corners = this.rectangleCornersFromBounds(this.editingData.layer.getBounds());
          this.editingData.points = corners;
          corners.forEach((corner, idx) => {if (this.editingMarkers[idx]) this.editingMarkers[idx].setLatLng(corner);});
        },
        updateEditingMarkerPositions() {if (!this.editingData) return; this.editingData.points.forEach((p, i) => {if (this.editingMarkers[i]) this.editingMarkers[i].setLatLng(p);});},

        buildPopup({label, type, latlng, meta = {}}) {
          const lines = [`<div class="font-semibold text-slate-100 text-sm mb-2">${label}</div>`];
          if (type === 'marker' && latlng) {
            lines.push(`<div class="text-xs text-slate-300">Lat: ${this.formatLat(latlng.lat)}</div>`);
            lines.push(`<div class="text-xs text-slate-300">Lng: ${this.formatLng(latlng.lng)}</div>`);
          } else if (type === 'polyline') {
            lines.push(`<div class="text-xs text-slate-300">Length: ${(meta.lengthKm || 0).toFixed(2)} km</div>`);
          } else if (type === 'polygon' || type === 'rectangle') {
            lines.push(`<div class="text-xs text-slate-300">Area: ${(meta.areaSqKm || 0).toFixed(2)} km²</div>`);
          } else if (type === 'connector') {
            const fromLabel = this.findOverlayLabel(meta.startId);
            const toLabel = this.findOverlayLabel(meta.endId);
            lines.push(`<div class="text-xs text-slate-300">Link: ${fromLabel} → ${toLabel}</div>`);
            lines.push(`<div class="text-xs text-slate-300">Style: ${meta.style || 'solid'} | Arrow: ${meta.direction || 'none'}</div>`);
            lines.push(`<div class="text-xs text-slate-300">Length: ${(meta.lengthKm || 0).toFixed(2)} km</div>`);
          }
          lines.push(`<div class="text-xs text-cyan-400 mt-2">Updated: ${new Date().toLocaleTimeString('en-GB', {hour12: false})}Z</div>`);
          return lines.join('');
        },

        refreshOverlayMeta() {
          const connectorsToRefresh = [];
          this.overlays = this.overlays.map((overlay) => {
            const layer = this.layerIndex[overlay.id];
            if (!layer) return overlay;
            if (overlay.type === 'marker' && layer.getLatLng) {
              const pos = layer.getLatLng();
              return {...overlay, lat: pos.lat, lng: pos.lng};
            }
            if (overlay.type === 'polyline') return {...overlay, ...this.calculatePolyline(layer)};
            if (overlay.type === 'polygon') return {...overlay, ...this.calculatePolygon(layer)};
            if (overlay.type === 'rectangle') return {...overlay, ...this.calculateRectangle(layer)};
            if (overlay.type === 'connector') {
              const metrics = this.calculatePolyline(layer);
              const start = this.getOverlayLatLng(overlay.startId);
              const end = this.getOverlayLatLng(overlay.endId);
              if (start) {metrics.startLat = start.lat; metrics.startLng = start.lng;}
              if (end) {metrics.endLat = end.lat; metrics.endLng = end.lng;}
              const next = {...overlay, ...metrics};
              this.applyConnectorStyling(layer, next);
              connectorsToRefresh.push(overlay.id);
              return next;
            }
            if (this.editingLayerId === overlay.id && this.editingData) {
              if (overlay.type === 'rectangle') this.updateRectangleMarkerPositions();
              else if (overlay.type === 'polyline' || overlay.type === 'polygon') {
                this.editingData.points = this.extractEditablePoints(layer, overlay.type);
                this.updateEditingMarkerPositions();
              }
            }
            return overlay;
          });
          connectorsToRefresh.forEach((connectorId) => {this.updateConnectorArrows(connectorId);});
        },

        calculatePolyline(layer) {
          const latlngs = layer.getLatLngs();
          if (!Array.isArray(latlngs) || latlngs.length < 2) return {lengthM: 0, lengthKm: 0, color: this.defaultColor('polyline')};
          let length = 0; for (let i = 0; i < latlngs.length - 1; i += 1) length += latlngs[i].distanceTo(latlngs[i + 1]);
          return {lengthM: length, lengthKm: length / 1000, color: this.defaultColor('polyline')};
        },

        calculatePolygon(layer) {
          const latlngs = layer.getLatLngs()[0] || [];
          const area = this.computePolygonArea(latlngs);
          return {areaSqM: area, areaSqKm: area / 1_000_000, color: this.defaultColor('polygon')};
        },

        calculateRectangle(layer) {
          const bounds = layer.getBounds();
          const corners = [
            bounds.getSouthWest(),
            L.latLng(bounds.getSouthWest().lat, bounds.getNorthEast().lng),
            bounds.getNorthEast(),
            L.latLng(bounds.getNorthEast().lat, bounds.getSouthWest().lng)
          ];
          const area = this.computePolygonArea(corners);
          return {areaSqM: area, areaSqKm: area / 1_000_000, color: this.defaultColor('rectangle')};
        },

        computePolygonArea(latlngs) {
          if (!latlngs || latlngs.length < 3) return 0;
          const radius = 6378137; let area = 0;
          for (let i = 0; i < latlngs.length; i += 1) {
            const p1 = latlngs[i]; const p2 = latlngs[(i + 1) % latlngs.length];
            const lon1 = this.toRad(p1.lng); const lon2 = this.toRad(p2.lng);
            const lat1 = this.toRad(p1.lat); const lat2 = this.toRad(p2.lat);
            area += (lon2 - lon1) * (2 + Math.sin(lat1) + Math.sin(lat2));
          }
          area = (area * radius * radius) / 2;
          return Math.abs(area);
        },

        normalizePolygonRings(latlngs) {
          const rings = [];
          const collect = (segment) => {
            if (!Array.isArray(segment) || !segment.length) return;
            const first = segment[0];
            if (first && typeof first.lat === 'number' && typeof first.lng === 'number') {
              rings.push(segment);
              return;
            }
            segment.forEach((child) => { collect(child); });
          };
          collect(latlngs);
          return rings;
        },

        isLatLngInsidePolygon(latlng, layer) {
          if (!latlng || !layer || typeof layer.getLatLngs !== 'function') return false;
          const rings = this.normalizePolygonRings(layer.getLatLngs());
          if (!rings.length) return false;
          if (!this.pointInRing(latlng, rings[0])) return false;
          for (let i = 1; i < rings.length; i += 1) {
            if (this.pointInRing(latlng, rings[i])) return false;
          }
          return true;
        },

        pointInRing(latlng, ring) {
          if (!Array.isArray(ring) || ring.length < 3) return false;
          const x = latlng.lng;
          const y = latlng.lat;
          let inside = false;
          for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
            const xi = ring[i].lng;
            const yi = ring[i].lat;
            const xj = ring[j].lng;
            const yj = ring[j].lat;
            const denominator = (yj - yi) || 1e-12;
            const intersects = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / denominator + xi);
            if (intersects) inside = !inside;
          }
          return inside;
        },

        toRad(deg) {return (deg * Math.PI) / 180;},

        defaultColor(type) {
          if (type === 'polyline') return '#f97316';
          if (type === 'polygon') return '#38bdf8';
          if (type === 'rectangle') return '#facc15';
          return '#38bdf8';
        },

        autoLabel(type) {
          const count = this.overlays.filter((o) => o.type === type).length + 1;
          if (type === 'marker') return `POINT ${String(count).padStart(3, '0')}`;
          if (type === 'polyline') return `ROUTE ${count}`;
          if (type === 'polygon') return `ZONE ${count}`;
          if (type === 'rectangle') return `BOX ${count}`;
          if (type === 'connector') return `LINK ${count}`;
          return `ENTITY ${count}`;
        },

        focusLayer(id) {
          const layer = this.layerIndex[id];
          if (!layer) return;
          this.selectedLayerId = id;
          if (layer.getBounds) this.map.fitBounds(layer.getBounds(), {padding: [40, 40]});
          else if (layer.getLatLng) {this.map.setView(layer.getLatLng(), Math.max(this.map.getZoom(), 13)); layer.openPopup();}
          this.updateStatus('Target focused', `Overlay ${id} centered on HUD.`);
        },

        confirmRemoveOverlay(id) {
          if (this.lockedOverlayId && this.lockedOverlayId !== id) return;
          if (id === null || id === undefined) return;
          if (this.skipDeleteConfirm) {
            this.removeOverlay(id);
            return;
          }
          const overlay = this.overlays.find((i) => i.id === id);
          this.pendingDeleteId = id;
          this.pendingDeleteLabel = overlay?.label || `Layer ${id}`;
          this.deleteConfirmDontAsk = false;
          this.lockedOverlayId = id;
          if (!this.layersOpen) this.layersOpen = true;
          this.closeFilterDialog();
          this.closeOverlapChooser();
        },

        cancelPendingDelete() {
          this.pendingDeleteId = null;
          this.pendingDeleteLabel = 'This overlay';
          this.deleteConfirmDontAsk = false;
          if (this.lockedOverlayId !== this.editingLayerId) this.lockedOverlayId = null;
        },

        executePendingDelete() {
          const id = this.pendingDeleteId;
          if (this.deleteConfirmDontAsk) this.skipDeleteConfirm = true;
          this.pendingDeleteId = null;
          this.deleteConfirmDontAsk = false;
          this.pendingDeleteLabel = 'This overlay';
          if (id === null || id === undefined) return;
          this.removeOverlay(id);
          this.lockedOverlayId = null;
        },

        removeOverlay(id) {
          const overlay = this.overlays.find((i) => i.id === id);
          const layer = this.layerIndex[id];
          const dependentConnectorIds = overlay && overlay.type !== 'connector'
            ? this.overlays.filter((i) => i.type === 'connector' && (i.startId === id || i.endId === id)).map((i) => i.id)
            : [];

          if (this.editingLayerId === id) this.stopEditing();
          if (overlay?.type === 'connector') this.removeConnectorArtifacts(id);
          layer?.closePopup?.();

          if (layer) {
            this.drawnItems.removeLayer?.(layer);
            this.groups.markers?.removeLayer?.(layer); // cluster
            this.groups.routes?.removeLayer?.(layer);
            this.groups.zones?.removeLayer?.(layer);
            this.groups.boxes?.removeLayer?.(layer);
            this.groups.connectors?.removeLayer?.(layer);
            layer.remove?.();
            delete this.layerIndex[id];
          }

          this.overlays = this.overlays.filter((i) => i.id !== id);
          if (this.selectedLayerId === id) this.selectedLayerId = null;
          dependentConnectorIds.forEach((cid) => {if (cid !== id) this.removeOverlay(cid);});

          this.updateStatus('Overlay removed', `Layer ${id} purged from map.`);
          if (this.pendingDeleteId === id) this.cancelPendingDelete();
          if (this.lockedOverlayId === id) this.lockedOverlayId = null;
          this.closeOverlapChooser();
        },

        clearAll() {
          this.stopEditing(); this.map?.closePopup?.();
          this.cancelPendingDelete();
          this.drawnItems?.clearLayers?.();
          this.groups.markers?.clearLayers?.(); // clears cluster
          this.groups.routes?.clearLayers?.();
          this.groups.zones?.clearLayers?.();
          this.groups.boxes?.clearLayers?.();
          this.groups.connectors?.clearLayers?.();
          Object.keys(this.connectorArtifacts).forEach((k) => {this.removeConnectorArtifacts(k);});
          this.connectorArtifacts = {}; this.layerIndex = {}; this.overlays = []; this.selectedLayerId = null;
          this.connectorForm = {startId: '', endId: '', style: 'solid', direction: 'none', anchorMode: 'auto'};
          this.resetDraft(this.isDrawingMode(this.mode) ? this.mode : null);
          this.updateStatus('Mission reset', 'All overlays cleared. Console sanitized.');
        },

        exportGeoJSON() {
          if (!this.drawnItems || !this.drawnItems.toGeoJSON) {this.updateStatus('Export unavailable', 'No overlays to export.'); return;}
          const data = this.drawnItems.toGeoJSON();
          const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/geo+json'});
          const url = URL.createObjectURL(blob);
          const anchor = document.createElement('a'); anchor.href = url; anchor.download = `atak-export-${new Date().toISOString()}.geojson`; anchor.click();
          URL.revokeObjectURL(url);
          this.updateStatus('Export complete', 'GeoJSON package ready for distribution.');
        },

        useDeviceLocation() {
          if (!navigator.geolocation) {this.updateStatus('Geolocation unsupported', 'Device does not expose secure positioning.'); return;}
          navigator.geolocation.getCurrentPosition(
            (position) => {
              const {latitude, longitude, accuracy} = position.coords;
              const latlng = L.latLng(latitude, longitude);
              this.map.setView(latlng, 15);
              this.injectMarker(latlng, {label: `OWN POS (${Math.round(accuracy)}m CEP)`, color: '#22c55e', symbol: '⬤', persist: true, skipReset: true});
              this.updateStatus('Own position locked', `Lat ${this.formatLat(latitude)}, Lng ${this.formatLng(longitude)}`);
            },
            () => {this.updateStatus('Position denied', 'User rejected geolocation request.');},
            {enableHighAccuracy: true, maximumAge: 10_000, timeout: 8_000}
          );
        },

        applyManualCoords() {
          if (!this.markerForm.manualLat || !this.markerForm.manualLng) {this.updateStatus('Coordinates missing', 'Latitude and longitude required.'); return;}
          const lat = Number.parseFloat(this.markerForm.manualLat);
          const lng = Number.parseFloat(this.markerForm.manualLng);
          if (Number.isNaN(lat) || Number.isNaN(lng)) {this.updateStatus('Invalid coordinates', 'Verify format and retry.'); return;}
          const latlng = L.latLng(lat, lng);
          this.injectMarker(latlng);
          this.map.setView(latlng, Math.max(this.map.getZoom(), 12));
          this.updateStatus('Manual marker deployed', `Lat ${this.formatLat(lat)}, Lng ${this.formatLng(lng)}`);
        },

        formatLat(value) {const h = value >= 0 ? 'N' : 'S'; return `${h} ${Math.abs(value).toFixed(5)}`;},
        formatLng(value) {const h = value >= 0 ? 'E' : 'W'; return `${h} ${Math.abs(value).toFixed(5)}`;},
        formatLength(meters) {if (!meters) return '0 m'; if (meters >= 1000) return `${(meters / 1000).toFixed(2)} km`; return `${meters.toFixed(0)} m`;},
        formatArea(squareMeters) {if (!squareMeters) return '0 m²'; if (squareMeters >= 1_000_000) return `${(squareMeters / 1_000_000).toFixed(2)} km²`; if (squareMeters >= 10_000) return `${(squareMeters / 10_000).toFixed(1)} ha`; return `${squareMeters.toFixed(0)} m²`;}
      };
    }
  </script>
</body>
</html>
"#;

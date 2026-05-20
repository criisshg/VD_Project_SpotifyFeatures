/* dashboard-boot.js
 *
 * Carrega les dades reals des de l'API Flask, les transforma a la forma que
 * espera el codi d'app.js (claus curtes per a tracks/tsne/pca, claus llargues
 * per a genreProfiles/clusterProfiles) i arrenca el dashboard manualment.
 *
 * S'injecta al final del <body> del template, després de l'app.js empaquetat.
 */

(async () => {
  // 1. Desactivar el boot automàtic d'app.js. El seu listener de
  //    DOMContentLoaded ja s'ha registrat però el flag el deixa inert.
  window.SKIP_APP_BOOT = true;

  const setStatus = (msg) => {
    const el = document.getElementById('__bundler_loading');
    if (el) el.textContent = msg;
  };
  setStatus('Loading data from /api/…');

  async function getJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
    return r.json();
  }

  let api;
  try {
    const [
      tracks, tsne, pca, clustersK3, clustersK5, clustersK7,
      genreProfiles, correlation,
      profilesK3, profilesK5, profilesK7,
      presetsJson, kpis,
    ] = await Promise.all([
      getJSON('/api/tracks'),
      getJSON('/api/tsne'),
      getJSON('/api/pca'),
      getJSON('/api/clusters?k=3'),
      getJSON('/api/clusters?k=5'),
      getJSON('/api/clusters?k=7'),
      getJSON('/api/genre-profiles'),
      getJSON('/api/correlation'),
      getJSON('/api/cluster-profiles?k=3'),
      getJSON('/api/cluster-profiles?k=5'),
      getJSON('/api/cluster-profiles?k=7'),
      getJSON('/api/presets'),
      getJSON('/api/kpis'),
    ]);
    api = {
      tracks, tsne, pca, clustersK3, clustersK5, clustersK7,
      genreProfiles, correlation,
      profilesByK: { 3: profilesK3, 5: profilesK5, 7: profilesK7 },
      presetsJson, kpis,
    };
  } catch (err) {
    setStatus('API error: ' + err.message);
    console.error('[dashboard-boot] fetch failed', err);
    return;
  }

  setStatus('Shaping data…');

  // 2. Tracks: claus llargues → claus curtes que espera app.js.
  //    Mapping confirmat al FEATS array d'app.js (línies ~1188-1201):
  //      p=popularity, ac=acousticness, da=danceability, du=duration_min,
  //      en=energy, ins=instrumentalness, li=liveness, lo=loudness,
  //      sp=speechiness, te=tempo, va=valence
  const tracksShort = api.tracks.map((t, i) => ({
    i,
    n: t.track_name,
    a: t.artist_name,
    g: t.genre,
    p: t.popularity,
    ac: t.acousticness,
    da: t.danceability,
    du: t.duration_min,
    en: t.energy,
    ins: t.instrumentalness,
    li: t.liveness,
    lo: t.loudness,
    sp: t.speechiness,
    te: t.tempo,
    va: t.valence,
  }));

  // 3. Indexar clusters per track_id → {C0,C1,…} segons k.
  const cluster = { k3: {}, k5: {}, k7: {} };
  for (const row of api.clustersK3) cluster.k3[row.track_id] = row.cluster_id;
  for (const row of api.clustersK5) cluster.k5[row.track_id] = row.cluster_id;
  for (const row of api.clustersK7) cluster.k7[row.track_id] = row.cluster_id;

  // 4. tsne points → forma curta amb cluster ids per cada k.
  const tsneShort = api.tsne.map(p => ({
    g: p.genre,
    x: p.tsne_x,
    y: p.tsne_y,
    p: p.popularity,
    tid: p.track_id,
    k3: cluster.k3[p.track_id] || 'C0',
    k5: cluster.k5[p.track_id] || 'C0',
    k7: cluster.k7[p.track_id] || 'C0',
  }));

  // 5. pca points → forma curta. Submostra els primers 8000 perquè Plotly
  //    no es mori amb 176k punts (Space és visualment idèntic).
  const PCA_LIMIT = 8000;
  const pcaSample = api.pca.length > PCA_LIMIT
    ? api.pca.filter((_, i) => i % Math.ceil(api.pca.length / PCA_LIMIT) === 0)
    : api.pca;
  const pcaShort = pcaSample.map(p => {
    const k3 = cluster.k3[p.track_id] || 'C0';
    const ci = parseInt(k3.replace(/\D/g, ''), 10) || 0;
    return {
      g: p.genre,
      x: p.PC1,
      y: p.PC2,
      c: ci,
      tid: p.track_id,
      k3,
      k5: cluster.k5[p.track_id] || 'C0',
      k7: cluster.k7[p.track_id] || 'C0',
    };
  });

  // 6. Vinyl sample: 5.000 tracks estratificat per gènere perquè el plot
  //    polar quedi llegible (el codi d'app.js usa D.tracksVinylSample si existeix).
  const VINYL_TARGET = 5000;
  const byGenre = {};
  for (const t of tracksShort) (byGenre[t.g] ||= []).push(t);
  const genres = Object.keys(byGenre);
  const perGenre = Math.max(1, Math.floor(VINYL_TARGET / genres.length));
  const vinylSample = [];
  for (const g of genres) {
    const list = byGenre[g];
    const step = Math.max(1, Math.floor(list.length / perGenre));
    for (let i = 0; i < list.length && vinylSample.length < VINYL_TARGET; i += step) {
      vinylSample.push(list[i]);
    }
  }

  // 7. Recompte per gènere → array de tuples [genre, count].
  const genreCounts = Object.entries(
    tracksShort.reduce((acc, t) => { acc[t.g] = (acc[t.g] || 0) + 1; return acc; }, {}),
  );

  // 8. clusterProfiles aplanat per al K actiu (per defecte k=3). Forma esperada
  //    pel fallback de renderClusterCards: {cluster_id: int, ...features llargues}.
  //    El listener del slider (pas 13) mutarà aquest array quan canviï K.
  const flattenProfiles = (profilesResp) =>
    (profilesResp.clusters || []).map((c, idx) => ({
      cluster_id: idx,
      ...c.centroid,
    }));
  const clusterProfilesFlat = flattenProfiles(api.profilesByK[3]);

  // 9. Llista de gèneres distintes (ordenada com al bundle: alfabètica).
  const genresList = Array.from(new Set(tracksShort.map(t => t.g))).sort();

  // 10. Features llargues, mateix ordre que el bundle.
  const featuresLong = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'speechiness', 'valence', 'popularity', 'tempo',
    'loudness', 'duration_min',
  ];

  // 11. Construir window.DATA. Important: app.js fa `const D = window.DATA;`
  //     així que NO podem reassignar window.DATA — perderiem la referència.
  //     Cal mutar l'objecte existent in-place.
  const newData = {
    features: featuresLong,
    genres: genresList,
    genreProfiles: api.genreProfiles,
    clusterProfiles: clusterProfilesFlat,
    tracks: tracksShort,
    tracksVinylSample: vinylSample,
    tsne: tsneShort,
    pca: pcaShort,
    genreCounts,
  };
  if (!window.DATA) window.DATA = {};
  for (const k of Object.keys(window.DATA)) {
    if (!(k in newData)) delete window.DATA[k];
  }
  Object.assign(window.DATA, newData);
  window.PLAYLIST_PRESETS = api.presetsJson;
  window.CLUSTER_PROFILES = api.profilesByK[3];
  window.__CLUSTER_PROFILES_BY_K = api.profilesByK;

  // PRECOMPUTED_CORR: el FeatureGraph fa corrMatrix[a][b] amb claus CURTES
  // (FEATS[i].key). Cal transformar {features, matrix} → objecte anidat.
  const longToShort = {
    popularity: 'p', acousticness: 'ac', danceability: 'da',
    duration_min: 'du', energy: 'en', instrumentalness: 'ins',
    liveness: 'li', loudness: 'lo', speechiness: 'sp',
    tempo: 'te', valence: 'va',
  };
  const fLong = api.correlation.features;
  const corrObj = {};
  fLong.forEach((a, i) => {
    const aShort = longToShort[a];
    if (!aShort) return;
    corrObj[aShort] = {};
    fLong.forEach((b, j) => {
      const bShort = longToShort[b];
      if (bShort) corrObj[aShort][bShort] = api.correlation.matrix[i][j];
    });
  });
  window.PRECOMPUTED_CORR = corrObj;

  // 12. Header stats (l'app.js ho fa al seu boot, però aquí el saltem).
  const setText = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  };
  setText('stat-tracks', api.kpis.tracks.toLocaleString('en-US'));
  setText('stat-genres', api.kpis.genres);
  setText('stat-features', api.kpis.features);

  setStatus('Rendering…');

  // 13. Cridar el .init() de cada mòdul d'app.js. Comparteixen script scope
  //    amb aquest boot.js (tots dos són scripts clàssics), així que els const
  //    top-level GenreBar/Builder/Vinyl/SonicSpace/Radar/FeatureGraph són
  //    visibles aquí.
  try {
    if (typeof GenreBar !== 'undefined')    GenreBar.init();
    if (typeof Builder !== 'undefined')     Builder.init();
    if (typeof Vinyl !== 'undefined')       Vinyl.init();
    if (typeof SonicSpace !== 'undefined')  SonicSpace.init();
    if (typeof Radar !== 'undefined')       Radar.init();
    if (typeof FeatureGraph !== 'undefined') FeatureGraph.init();
  } catch (err) {
    setStatus('Init error: ' + err.message);
    console.error('[dashboard-boot] init failed', err);
    return;
  }

  // 14. Sincronitzar profiles quan canvia el slider K. El listener corre en
  //     fase de capture per assegurar que window.CLUSTER_PROFILES i
  //     D.clusterProfiles estan actualitzats ABANS que el listener intern
  //     d'app.js cridi renderClusterCards().
  document.querySelectorAll('#k-slider-track .stop').forEach(stop => {
    stop.addEventListener('click', () => {
      const k = parseInt(stop.dataset.k, 10);
      const profs = window.__CLUSTER_PROFILES_BY_K[k];
      if (!profs) return;
      window.CLUSTER_PROFILES = profs;
      // Mutar D.clusterProfiles in-place: l'app.js manté la referència via `const D = window.DATA`.
      window.DATA.clusterProfiles.length = 0;
      profs.clusters.forEach((c, idx) => {
        window.DATA.clusterProfiles.push({ cluster_id: idx, ...c.centroid });
      });
    }, true);
  });

  // 15. Amagar l'indicador de càrrega del bundler.
  const loading = document.getElementById('__bundler_loading');
  if (loading) loading.remove();
})();

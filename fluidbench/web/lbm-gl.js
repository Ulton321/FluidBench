/* D2Q9 lattice Boltzmann on the GPU, in WebGL2.
 *
 * IMPORTANT: this is *not* the kernel FluidBench times.  The benchmark runs
 * the Python solver in fluidbench/lbm.py; this is a separate reimplementation
 * whose only job is to put the flow on screen at 60 fps.  Treating this as
 * the benchmarked workload would be exactly the mistake the project exists
 * to argue against.
 *
 * It does follow the same step order, so what you see is the same physics:
 *
 *   1. stream   -- pull f_i from the neighbour at -c_i (periodic, so a wrap)
 *   2. capture  -- the populations that landed in the cylinder, reversed
 *   3. reduce   -- density and velocity
 *   4. collide  -- BGK relaxation towards the local equilibrium
 *   5. bounce   -- write the reversed populations back
 *
 * State lives in three RGBA32F textures (f0..f3, f4..f7, f8) plus a fourth
 * attachment carrying the macroscopic field for the renderer, all written in
 * one pass with MRT and ping-ponged between two sets.
 *
 * Half float is not an option here: rho is around 100 and a step changes a
 * population by ~1e-4, which sits far below the 10-bit mantissa.  Without
 * EXT_color_buffer_float there is no simulation, and the page falls back to
 * a still image.
 */

const NL = 9;

// Matching fluidbench/lbm.py: index -> (cx, cy), and the reversed direction.
const CXS = [0, 0, 1, 1, 1, 0, -1, -1, -1];
const CYS = [0, 1, 1, 0, -1, -1, -1, 0, 1];

const VERT = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

// Shared preamble: the lattice, and the equilibrium the collision relaxes to.
const LATTICE = `
const vec2 C[9] = vec2[9](
  vec2( 0.0, 0.0), vec2( 0.0, 1.0), vec2( 1.0, 1.0), vec2( 1.0, 0.0),
  vec2( 1.0,-1.0), vec2( 0.0,-1.0), vec2(-1.0,-1.0), vec2(-1.0, 0.0),
  vec2(-1.0, 1.0));
const float W[9] = float[9](
  4.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0,
  1.0/9.0, 1.0/36.0, 1.0/9.0, 1.0/36.0);
const int OPP[9] = int[9](0, 5, 6, 7, 8, 1, 2, 3, 4);

float feq(float rho, vec2 u, int i) {
  float cu  = dot(C[i], u);
  float usq = dot(u, u);
  return rho * W[i] * (1.0 + 3.0*cu + 4.5*cu*cu - 1.5*usq);
}
ivec2 wrap(ivec2 p, ivec2 n) { return (p % n + n) % n; }
`;

const STEP_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uF0, uF1, uF2;
uniform vec2  uGrid;
uniform float uInvTau;
uniform vec2  uCyl;
uniform float uRadius;
uniform float uInflow;
uniform float uRho0;
uniform float uInlet;    // inlet strip width in cells; 0.0 = fully periodic
uniform float uPhase;    // advances once per step, for the inlet jitter

layout(location=0) out vec4 oF0;
layout(location=1) out vec4 oF1;
layout(location=2) out vec4 oF2;
layout(location=3) out vec4 oMacro;
${LATTICE}

float fetch(int i, ivec2 p) {
  if (i < 4) return texelFetch(uF0, p, 0)[i];
  if (i < 8) return texelFetch(uF1, p, 0)[i - 4];
  return texelFetch(uF2, p, 0).r;
}

float hash(vec2 v) {
  return fract(sin(dot(v, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  ivec2 n = ivec2(uGrid);
  ivec2 p = ivec2(gl_FragCoord.xy);

  // 1. Streaming, pull form: f_i arrives from the cell one step back along c_i.
  //    texelFetch takes no derivatives and wrap() makes the domain periodic,
  //    which is what np.roll does in the Python solver.
  float f[9];
  for (int i = 0; i < 9; i++) {
    f[i] = fetch(i, wrap(p - ivec2(C[i]), n));
  }

  // 3. Macroscopic moments of the streamed populations.
  float rho = 0.0;
  vec2  mom = vec2(0.0);
  for (int i = 0; i < 9; i++) { rho += f[i]; mom += f[i] * C[i]; }
  vec2 u = mom / max(rho, 1e-6);

  vec2  d     = vec2(p) - uCyl;
  bool  solid = dot(d, d) < uRadius * uRadius;
  bool  inlet = uInlet > 0.5 && float(p.x) < uInlet;

  float out9[9];
  if (solid) {
    // 2 + 5. Bounce-back: the reversed streamed population, and collision
    //        never touches this cell -- the same order as the Python step.
    for (int i = 0; i < 9; i++) out9[i] = f[OPP[i]];
    rho = uRho0;
    u   = vec2(0.0);
  } else if (inlet) {
    // A driven inlet, which the Python solver does not have: it is fully
    // periodic and unforced, so its mean velocity decays as the cylinder
    // takes momentum out.  That is fine for a few thousand timed steps and
    // wrong for a background that has to run all afternoon.
    float jitter = (hash(vec2(float(p.y), uPhase)) - 0.5) * 0.06;
    u   = vec2(uInflow * (1.0 + jitter), uInflow * jitter);
    rho = uRho0;
    for (int i = 0; i < 9; i++) out9[i] = feq(rho, u, i);
  } else {
    // 4. BGK collision.
    for (int i = 0; i < 9; i++) {
      out9[i] = f[i] - uInvTau * (f[i] - feq(rho, u, i));
    }
  }

  oF0 = vec4(out9[0], out9[1], out9[2], out9[3]);
  oF1 = vec4(out9[4], out9[5], out9[6], out9[7]);
  oF2 = vec4(out9[8], 0.0, 0.0, 0.0);
  oMacro = vec4(rho, u.x, u.y, solid ? 1.0 : 0.0);
}`;

// Vorticity, shared by the renderer and the auto-scale reduction so the two
// never disagree about what they are measuring.  Central differences, exactly
// as LBMSolver.vorticity() takes them.
const VORTICITY = `
float vorticity(sampler2D macro, ivec2 p, ivec2 n) {
  float uxUp   = texelFetch(macro, wrap(p + ivec2(0,  1), n), 0).y;
  float uxDown = texelFetch(macro, wrap(p + ivec2(0, -1), n), 0).y;
  float uyR    = texelFetch(macro, wrap(p + ivec2( 1, 0), n), 0).z;
  float uyL    = texelFetch(macro, wrap(p + ivec2(-1, 0), n), 0).z;
  return (uyR - uyL) * 0.5 - (uxUp - uxDown) * 0.5;
}`;

const DRAW_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uMacro;
uniform vec2  uGrid;
uniform vec2  uView;
uniform float uLimit;
uniform float uSpeedLimit;
uniform int   uMap;
uniform float uDim;
uniform vec2  uCyl;
uniform float uRadius;
out vec4 oColor;
${LATTICE}
${VORTICITY}

vec3 ramp(vec3 a, vec3 b, vec3 mid, vec3 c, vec3 d, float t) {
  // t in [-1, 1]; two hues meeting at a neutral midpoint, equal steps per arm.
  float s = abs(t) * 2.0;
  vec3 lo = t < 0.0 ? b : c;
  vec3 hi = t < 0.0 ? a : d;
  return s < 1.0 ? mix(mid, lo, s) : mix(lo, hi, s - 1.0);
}

// Five-stop sequential scale for speed: deep navy through blue and cyan to
// a warm off-white, so the fast free stream reads bright and the wake dark.
vec3 sequential(float s) {
  s = clamp(s, 0.0, 1.0) * 4.0;
  vec3 c0 = vec3(0.035, 0.043, 0.075);
  vec3 c1 = vec3(0.090, 0.180, 0.380);
  vec3 c2 = vec3(0.157, 0.435, 0.753);
  vec3 c3 = vec3(0.373, 0.765, 0.839);
  vec3 c4 = vec3(0.945, 0.957, 0.890);
  if (s < 1.0) return mix(c0, c1, s);
  if (s < 2.0) return mix(c1, c2, s - 1.0);
  if (s < 3.0) return mix(c2, c3, s - 2.0);
  return mix(c3, c4, s - 3.0);
}

vec3 colormap(float t) {
  if (uMap == 1) {
    return ramp(vec3(0.369,0.918,0.831), vec3(0.098,0.620,0.439),
                vec3(0.110,0.114,0.125),
                vec3(0.788,0.522,0.000), vec3(0.984,0.816,0.478), t);
  }
  if (uMap == 2) {
    float s = clamp(abs(t), 0.0, 1.0);
    return mix(vec3(0.035,0.043,0.075), vec3(0.620,0.773,0.957), pow(s, 0.75));
  }
  return ramp(vec3(0.620,0.773,0.957), vec3(0.224,0.529,0.898),
              vec3(0.110,0.114,0.125),
              vec3(0.890,0.286,0.282), vec3(0.961,0.639,0.635), t);
}

// The scalar being shown at one lattice cell.
float sampleField(ivec2 p, ivec2 n) {
  if (uMap == 3) {
    vec4 m = texelFetch(uMacro, p, 0);
    return length(m.yz);
  }
  return vorticity(uMacro, p, n);
}

void main() {
  ivec2 n  = ivec2(uGrid);
  vec2  uv = gl_FragCoord.xy / uView;
  vec2  g  = uv * uGrid;

  // Bilinear reconstruction between cell centres, so a coarse lattice on a
  // large screen reads as a continuous field rather than a mosaic.
  vec2  q  = g - 0.5;
  ivec2 p0 = ivec2(floor(q));
  vec2  fr = q - floor(q);
  float a = sampleField(wrap(p0,               n), n);
  float b = sampleField(wrap(p0 + ivec2(1, 0), n), n);
  float c = sampleField(wrap(p0 + ivec2(0, 1), n), n);
  float d = sampleField(wrap(p0 + ivec2(1, 1), n), n);
  float v = mix(mix(a, b, fr.x), mix(c, d, fr.x), fr.y);

  vec3 col = uMap == 3
    ? sequential(v / max(uSpeedLimit, 1e-9))
    : colormap(clamp(v / max(uLimit, 1e-9), -1.0, 1.0));

  // The obstacle, drawn analytically with an anti-aliased rim so it stays
  // crisp at any resolution and follows the pointer even while paused.
  float px   = uGrid.y / uView.y;                  // lattice cells per pixel
  float dist = length(g - uCyl) - uRadius;
  float body = 1.0 - smoothstep(-px, px, dist);
  float rim  = 1.0 - smoothstep(0.0, 2.0 * px, abs(dist + px));
  vec3  solid = vec3(0.078, 0.086, 0.106);
  col = mix(col, solid, body);
  col = mix(col, vec3(0.86, 0.88, 0.92), rim * 0.55);

  oColor = vec4(col * uDim, 1.0);
}`;

// Block-average of |vorticity|, read back once every few seconds to set the
// colour limit -- the same job _symmetric_limit() does on the host in
// visualize.py, and the same reason: a scale taken from one frame either
// saturates later ones or washes them out.
const REDUCE_FRAG = `#version 300 es
precision highp float;
precision highp sampler2D;

uniform sampler2D uMacro;
uniform vec2 uGrid;
uniform vec2 uOut;
out vec4 oColor;
${LATTICE}
${VORTICITY}

void main() {
  ivec2 n = ivec2(uGrid);
  vec2 block = uGrid / uOut;
  vec2 origin = gl_FragCoord.xy - vec2(0.5);

  float sum = 0.0, peak = 0.0, bad = 0.0, count = 0.0;
  for (int j = 0; j < 4; j++) {
    for (int i = 0; i < 4; i++) {
      vec2 at = (origin + (vec2(float(i), float(j)) + 0.5) * 0.25) * block;
      ivec2 p = wrap(ivec2(at), n);
      if (texelFetch(uMacro, p, 0).w > 0.5) continue;     // skip the cylinder
      float rho = texelFetch(uMacro, p, 0).x;
      float w = vorticity(uMacro, p, n);
      // A blown-up run fails both of these; NaN fails every comparison.
      if (!(rho > 1.0 && rho < 1.0e4) || !(abs(w) < 1.0e3)) { bad += 1.0; continue; }
      sum += abs(w);
      peak = max(peak, abs(w));
      count += 1.0;
    }
  }
  oColor = vec4(sum, peak, bad, count);
}`;

/* ---------------------------------------------------------------------- */

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  // Box-Muller; the Python side uses numpy's generator, so the streams differ.
  // Only the statistics need to match -- this is jitter to break the wake's
  // symmetry, not an initial condition anyone compares against.
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`shader failed to compile: ${log}`);
  }
  return shader;
}

function program(gl, fragSource) {
  const prog = gl.createProgram();
  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragSource);
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.bindAttribLocation(prog, 0, 'aPos');
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error(`program failed to link: ${log}`);
  }
  const uniforms = {};
  const count = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const { name } = gl.getActiveUniform(prog, i);
    uniforms[name] = gl.getUniformLocation(prog, name);
  }
  return { prog, u: uniforms };
}

export const COLORMAPS = [
  { id: 0, name: 'Vorticity', note: 'Signed curl of velocity. Blue is clockwise, red counter-clockwise.',
    gradient: 'linear-gradient(90deg,#9ec5f4,#3987e5,#1c1d20,#e34948,#f5a3a2)', labels: ['CW', '0', 'CCW'] },
  { id: 1, name: 'Vorticity (warm)', note: 'Signed curl of velocity, teal and amber poles.',
    gradient: 'linear-gradient(90deg,#5eead4,#199e70,#1c1d20,#c98500,#fbd07a)', labels: ['CW', '0', 'CCW'] },
  { id: 2, name: 'Vorticity magnitude', note: 'Rotation strength regardless of direction.',
    gradient: 'linear-gradient(90deg,#090b13,#9ec5f4)', labels: ['0', '', 'max'] },
  { id: 3, name: 'Speed', note: 'Velocity magnitude |u|, relative to the free stream.',
    gradient: 'linear-gradient(90deg,#090b13,#172e61,#286fc0,#5fc3d6,#f1f4e3)', labels: ['0', '', '1.6 U∞'] },
];

export const REDUCE_W = 48;
export const REDUCE_H = 24;

export class FlowField {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} options  tau, inflow, seed, rows, sustained, spinup
   */
  constructor(canvas, options = {}) {
    this.canvas = canvas;
    this.opts = {
      tau: 0.53,
      inflow: 0.1,
      rho0: 100,
      noise: 0.01,
      seed: 42,
      rows: 176,
      maxCols: 760,
      sustained: true,
      // The wake needs several thousand steps before it rolls up -- the CLI
      // fast-forwards 4000 for the same reason.  A widescreen background
      // wants more than that: the street only reads as a street once a few
      // vortices have detached and travelled downstream.
      spinup: 11000,
      stepsPerFrame: 5,
      // Slimmer than the solver's 13% default, and set further upstream.
      // The reference case is 4:1, where a fat cylinder still leaves eleven
      // diameters of wake; a 16:9 window does not, and a street with one
      // vortex in it is just a blob.
      radiusFraction: 0.10,
      cylinderAt: 0.2,
      colormap: 0,
      contrast: 1,
      dim: 1,
      ...options,
    };

    this.gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!this.gl) throw new Error('WebGL2 is not available in this browser');
    if (!this.gl.getExtension('EXT_color_buffer_float')) {
      throw new Error(
        'EXT_color_buffer_float is missing, so the simulation cannot use the ' +
        'float textures it needs'
      );
    }

    this.steps = 0;
    this.limit = 0.02;
    this.limitSettled = false;
    this.diverged = false;
    this.phase = 0;
    this.running = false;
    this.spinning = false;
    this.spinDone = 0;
    this.frameHandle = 0;
    this.lastReduce = 0;
    this.onStatus = null;

    this._setup();
    this.reset();
  }

  /** The GPU the *browser* chose, which on a laptop is often not the one
   *  torch is benchmarking.  Worth saying out loud: if the browser picked
   *  the same card, this animation would be contending with the device
   *  under test. */
  renderer() {
    const info = this.gl.getExtension('WEBGL_debug_renderer_info');
    if (!info) return this.gl.getParameter(this.gl.RENDERER) || 'unknown GPU';
    return this.gl.getParameter(info.UNMASKED_RENDERER_WEBGL) || 'unknown GPU';
  }

  /* -- GL objects ------------------------------------------------------- */

  _setup() {
    const gl = this.gl;
    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW);

    this.step = program(gl, STEP_FRAG);
    this.draw = program(gl, DRAW_FRAG);
    this.reduce = program(gl, REDUCE_FRAG);

    this.reduceTex = this._texture(REDUCE_W, REDUCE_H, gl.RGBA32F);
    this.reduceFbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.reduceFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D,
      this.reduceTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.reduceBuf = new Float32Array(REDUCE_W * REDUCE_H * 4);

    this.state = null;
  }

  _texture(width, height, format) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, format, width, height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  _releaseState() {
    if (!this.state) return;
    const gl = this.gl;
    for (const side of this.state.sides) {
      side.textures.forEach((t) => gl.deleteTexture(t));
      gl.deleteFramebuffer(side.fbo);
    }
    gl.deleteTexture(this.state.macro);
    this.state = null;
  }

  /* -- lifecycle -------------------------------------------------------- */

  /** Pick a grid that matches the canvas aspect, then seed it at equilibrium. */
  reset(sizeHint) {
    const gl = this.gl;
    const cssW = sizeHint?.width || this.canvas.clientWidth || 960;
    const cssH = sizeHint?.height || this.canvas.clientHeight || 540;
    const aspect = Math.max(0.2, cssW / Math.max(cssH, 1));

    const ny = Math.max(48, Math.round(this.opts.rows));
    const nx = Math.min(this.opts.maxCols, Math.max(96, Math.round(ny * aspect)));

    this._releaseState();

    const sides = [0, 1].map(() => {
      const textures = [
        this._texture(nx, ny, gl.RGBA32F),
        this._texture(nx, ny, gl.RGBA32F),
        this._texture(nx, ny, gl.RGBA32F),
      ];
      return { textures, fbo: gl.createFramebuffer() };
    });
    const macro = this._texture(nx, ny, gl.RGBA32F);

    // Both ping-pong targets write the macroscopic field to the same texture:
    // only the newest step's copy is ever read, so one is enough.
    for (const side of sides) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, side.fbo);
      side.textures.forEach((tex, i) => {
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i,
          gl.TEXTURE_2D, tex, 0);
      });
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3,
        gl.TEXTURE_2D, macro, 0);
      // drawBuffers is per-framebuffer state, so it belongs here rather than
      // next to the draw call: set against the wrong binding it silently
      // leaves attachments 1..3 unwritten, and the run is over before it
      // starts with nothing but a black screen to say so.
      gl.drawBuffers([
        gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1,
        gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3,
      ]);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error(`framebuffer is incomplete (0x${status.toString(16)})`);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.state = { nx, ny, macro, sides, front: 0 };
    this.radius = this.opts.radiusFraction * ny;
    // Half a cell off centre.  The solver leaves the cylinder exactly centred
    // and waits for the seeded noise to break the symmetry, which takes
    // thousands of steps; a real cylinder is never centred to the micron, and
    // the imperfection gets the street started sooner without changing what
    // the flow ends up doing.
    this.centre = [Math.floor(nx * this.opts.cylinderAt), ny / 2 + 0.5];

    this._seed();
    this.steps = 0;
    this.phase = 0;
    this.diverged = false;
    this.limitSettled = false;
    this.limit = Math.max(1e-4, (2.2 * this.opts.inflow) / this.radius);
    this.spinDone = 0;
    this.spinning = this.opts.spinup > 0;
    this._emitStatus();
    return this.state;
  }

  /** Upload populations at local equilibrium, as initial_distribution() does. */
  _seed() {
    const gl = this.gl;
    const { nx, ny, sides } = this.state;
    const { inflow, noise, rho0 } = this.opts;
    const rand = mulberry32(this.opts.seed);

    const planes = [
      new Float32Array(nx * ny * 4),
      new Float32Array(nx * ny * 4),
      new Float32Array(nx * ny * 4),
    ];

    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        // Starting *at* equilibrium, not at f = 1 + noise.  The tutorial
        // version of that mistake over-relaxes on the first step and blows
        // up inside a minute -- see the README.
        const ux = inflow * (1 + noise * gaussian(rand));
        const uy = inflow * noise * gaussian(rand);
        const usq = ux * ux + uy * uy;
        const base = (y * nx + x) * 4;
        for (let i = 0; i < NL; i++) {
          const cu = ux * CXS[i] + uy * CYS[i];
          const w = i === 0 ? 4 / 9 : (CXS[i] !== 0 && CYS[i] !== 0) ? 1 / 36 : 1 / 9;
          const value = rho0 * w * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq);
          planes[Math.floor(i / 4)][base + (i % 4)] = value;
        }
      }
    }

    for (const side of sides) {
      side.textures.forEach((tex, i) => {
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, nx, ny, gl.RGBA, gl.FLOAT, planes[i]);
      });
    }
  }

  reseed() {
    this.opts.seed = (this.opts.seed * 1664525 + 1013904223) >>> 0;
    this.reset();
  }

  setParams(changes = {}) {
    const needsReset = ['rows', 'seed', 'maxCols'].some(
      (key) => key in changes && changes[key] !== this.opts[key]);
    Object.assign(this.opts, changes);
    if (needsReset) this.reset();
    if ('inflow' in changes || 'tau' in changes) this.limitSettled = false;
    if ('radiusFraction' in changes && this.state) {
      this.radius = this.opts.radiusFraction * this.state.ny;
    }
    if (!this.running && this.state && !needsReset) this.render();
    this._emitStatus();
  }

  /* -- the loop --------------------------------------------------------- */

  _bindQuad(prog) {
    const gl = this.gl;
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  }

  /** One lattice step: render the whole grid through the step shader. */
  advance(count = 1) {
    const gl = this.gl;
    const { nx, ny } = this.state;
    const { prog, u } = this.step;

    this._bindQuad(prog);
    gl.viewport(0, 0, nx, ny);
    gl.uniform2f(u.uGrid, nx, ny);
    gl.uniform1f(u.uInvTau, 1 / this.opts.tau);
    gl.uniform2f(u.uCyl, this.centre[0], this.centre[1]);
    gl.uniform1f(u.uRadius, this.radius);
    gl.uniform1f(u.uInflow, this.opts.inflow);
    gl.uniform1f(u.uRho0, this.opts.rho0);
    gl.uniform1f(u.uInlet, this.opts.sustained ? 2 : 0);
    gl.uniform1i(u.uF0, 0);
    gl.uniform1i(u.uF1, 1);
    gl.uniform1i(u.uF2, 2);

    for (let k = 0; k < count; k++) {
      const src = this.state.sides[this.state.front];
      const dst = this.state.sides[1 - this.state.front];
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
      for (let i = 0; i < 3; i++) {
        gl.activeTexture(gl.TEXTURE0 + i);
        gl.bindTexture(gl.TEXTURE_2D, src.textures[i]);
      }
      this.phase = (this.phase + 1) % 1024;
      gl.uniform1f(u.uPhase, this.phase);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.state.front = 1 - this.state.front;
      this.steps++;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Block-average |vorticity| and pull back 48x24 floats to set the scale. */
  calibrate() {
    const gl = this.gl;
    const { nx, ny, macro } = this.state;
    const { prog, u } = this.reduce;

    this._bindQuad(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.reduceFbo);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, REDUCE_W, REDUCE_H);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, macro);
    gl.uniform1i(u.uMacro, 0);
    gl.uniform2f(u.uGrid, nx, ny);
    gl.uniform2f(u.uOut, REDUCE_W, REDUCE_H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.readPixels(0, 0, REDUCE_W, REDUCE_H, gl.RGBA, gl.FLOAT, this.reduceBuf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    let sum = 0, bad = 0, count = 0, peak = 0;
    for (let i = 0; i < this.reduceBuf.length; i += 4) {
      sum += this.reduceBuf[i];
      peak = Math.max(peak, this.reduceBuf[i + 1]);
      bad += this.reduceBuf[i + 2];
      count += this.reduceBuf[i + 3];
    }

    // More than a fifth of the field non-finite or absurd is a blow-up, not
    // a transient.  Report it the way `bench` reports a diverged run rather
    // than quietly papering over it.
    if (count < 1 || bad > count * 0.25 || !Number.isFinite(sum)) {
      this.diverged = true;
      this._emitStatus();
      return;
    }

    const mean = sum / count;
    // The mean over the whole field sits well under the wake's own scale;
    // this factor lands the colour limit near the 99.5th percentile the
    // Python renderer uses, checked against the same flow at 400x100.
    const target = Math.max(1e-5, mean * 5.5, peak * 0.02);
    this.limit = this.limitSettled ? this.limit * 0.85 + target * 0.15 : target;
    this.limitSettled = true;
  }

  render() {
    const gl = this.gl;
    const { nx, ny, macro } = this.state;
    const { prog, u } = this.draw;
    const width = this.canvas.width;
    const height = this.canvas.height;

    this._bindQuad(prog);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, macro);
    gl.uniform1i(u.uMacro, 0);
    gl.uniform2f(u.uGrid, nx, ny);
    gl.uniform2f(u.uView, width, height);
    gl.uniform1f(u.uLimit, this.limit / Math.max(0.15, this.opts.contrast));
    gl.uniform1f(u.uSpeedLimit,
      (1.6 * this.opts.inflow) / Math.max(0.15, this.opts.contrast));
    gl.uniform1i(u.uMap, this.opts.colormap | 0);
    gl.uniform1f(u.uDim, this.opts.dim);
    gl.uniform2f(u.uCyl, this.centre[0], this.centre[1]);
    gl.uniform1f(u.uRadius, this.radius);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Move the obstacle, in fractions of the domain (0..1 on each axis). */
  setObstacle(fx, fy) {
    if (!this.state) return;
    const { nx, ny } = this.state;
    const r = this.radius + 2;
    this.opts.cylinderAt = Math.min(1, Math.max(0, fx));
    this.centre = [
      Math.min(nx - r, Math.max(r, fx * nx)),
      Math.min(ny - r, Math.max(r, fy * ny)),
    ];
    if (!this.running) this.render();
  }

  /** Obstacle centre in fractions of the domain, and its radius in rows. */
  obstacle() {
    const { nx, ny } = this.state;
    return { x: this.centre[0] / nx, y: this.centre[1] / ny, r: this.radius / ny };
  }

  resize(width, height, dpr = 1) {
    const w = Math.max(1, Math.round(width * dpr));
    const h = Math.max(1, Math.round(height * dpr));
    if (this.canvas.width === w && this.canvas.height === h) return false;
    this.canvas.width = w;
    this.canvas.height = h;
    return true;
  }

  _emitStatus() {
    this.onStatus?.({
      steps: this.steps,
      grid: this.state ? `${this.state.nx}×${this.state.ny}` : '',
      cells: this.state ? this.state.nx * this.state.ny : 0,
      limit: this.limit,
      diverged: this.diverged,
      spinning: this.spinning,
      spinProgress: this.opts.spinup ? this.spinDone / this.opts.spinup : 1,
      running: this.running,
      reynolds: (this.opts.inflow * 2 * this.radius) / ((this.opts.tau - 0.5) / 3),
    });
  }

  _frame = () => {
    if (!this.running) return;
    this.frameHandle = requestAnimationFrame(this._frame);
    if (this.diverged) { this.render(); return; }

    if (this.spinning) {
      // Fast-forward to a developed wake before anyone looks at it, the same
      // reason cmd_sim defaults to --spinup 4000: the first few thousand
      // steps of a near-uniform field look like a hung program.
      const chunk = 300;
      this.advance(chunk);
      this.spinDone += chunk;
      if (this.spinDone >= this.opts.spinup) {
        this.spinning = false;
        this.calibrate();
      }
      this._emitStatus();
    } else {
      this.advance(this.opts.stepsPerFrame);
      const now = performance.now();
      if (now - this.lastReduce > 1800) {
        this.lastReduce = now;
        this.calibrate();
        this._emitStatus();
      }
    }
    this.render();
  };

  start() {
    if (this.running) return;
    this.running = true;
    this.lastReduce = performance.now();
    this.frameHandle = requestAnimationFrame(this._frame);
    this._emitStatus();
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.frameHandle);
    this._emitStatus();
  }

  /** Run to a developed wake and draw one frame, without animating.
   *  Used when the viewer has asked for reduced motion: they still get the
   *  shed street, just held still rather than looping. */
  renderStill() {
    this.advance(this.opts.spinup);
    this.spinning = false;
    this.calibrate();
    this.render();
    this._emitStatus();
  }

  destroy() {
    this.stop();
    this._releaseState();
    const gl = this.gl;
    gl.deleteTexture(this.reduceTex);
    gl.deleteFramebuffer(this.reduceFbo);
    gl.deleteBuffer(this.quad);
    [this.step, this.draw, this.reduce].forEach((p) => gl.deleteProgram(p.prog));
  }
}

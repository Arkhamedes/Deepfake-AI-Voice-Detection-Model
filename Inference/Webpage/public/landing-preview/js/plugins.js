
/*!
 * simpleParallax.min - simpleParallax is a simple JavaScript library that gives your website parallax animations on any images, 
 * @date: 01-02-2020 23:1:26, 
 * @version: 5.3.0,
 * @link: https://simpleparallax.com/
 */
!function (t, e) { "object" == typeof exports && "object" == typeof module ? module.exports = e() : "function" == typeof define && define.amd ? define("simpleParallax", [], e) : "object" == typeof exports ? exports.simpleParallax = e() : t.simpleParallax = e() }(window, (function () { return function (t) { var e = {}; function n(i) { if (e[i]) return e[i].exports; var s = e[i] = { i: i, l: !1, exports: {} }; return t[i].call(s.exports, s, s.exports, n), s.l = !0, s.exports } return n.m = t, n.c = e, n.d = function (t, e, i) { n.o(t, e) || Object.defineProperty(t, e, { enumerable: !0, get: i }) }, n.r = function (t) { "undefined" != typeof Symbol && Symbol.toStringTag && Object.defineProperty(t, Symbol.toStringTag, { value: "Module" }), Object.defineProperty(t, "__esModule", { value: !0 }) }, n.t = function (t, e) { if (1 & e && (t = n(t)), 8 & e) return t; if (4 & e && "object" == typeof t && t && t.__esModule) return t; var i = Object.create(null); if (n.r(i), Object.defineProperty(i, "default", { enumerable: !0, value: t }), 2 & e && "string" != typeof t) for (var s in t) n.d(i, s, function (e) { return t[e] }.bind(null, s)); return i }, n.n = function (t) { var e = t && t.__esModule ? function () { return t.default } : function () { return t }; return n.d(e, "a", e), e }, n.o = function (t, e) { return Object.prototype.hasOwnProperty.call(t, e) }, n.p = "", n(n.s = 0) }([function (t, e, n) { "use strict"; function i(t, e) { for (var n = 0; n < e.length; n++) { var i = e[n]; i.enumerable = i.enumerable || !1, i.configurable = !0, "value" in i && (i.writable = !0), Object.defineProperty(t, i.key, i) } } n.r(e); var s = new (function () { function t() { !function (t, e) { if (!(t instanceof e)) throw new TypeError("Cannot call a class as a function") }(this, t), this.positions = { top: 0, bottom: 0, height: 0 } } var e, n, s; return e = t, (n = [{ key: "setViewportTop", value: function (t) { return this.positions.top = t ? t.scrollTop : window.pageYOffset, this.positions } }, { key: "setViewportBottom", value: function () { return this.positions.bottom = this.positions.top + this.positions.height, this.positions } }, { key: "setViewportAll", value: function (t) { return this.positions.top = t ? t.scrollTop : window.pageYOffset, this.positions.height = t ? t.clientHeight : document.documentElement.clientHeight, this.positions.bottom = this.positions.top + this.positions.height, this.positions } }]) && i(e.prototype, n), s && i(e, s), t }()), o = function (t) { return NodeList.prototype.isPrototypeOf(t) || HTMLCollection.prototype.isPrototypeOf(t) ? Array.from(t) : "string" == typeof t || t instanceof String ? document.querySelectorAll(t) : [t] }, r = function () { for (var t, e = "transform webkitTransform mozTransform oTransform msTransform".split(" "), n = 0; void 0 === t;)t = void 0 !== document.createElement("div").style[e[n]] ? e[n] : void 0, n += 1; return t }(), a = function (t) { return !!t && (!!t.complete && (void 0 === t.naturalWidth || 0 !== t.naturalWidth)) }; function l(t) { return function (t) { if (Array.isArray(t)) { for (var e = 0, n = new Array(t.length); e < t.length; e++)n[e] = t[e]; return n } }(t) || function (t) { if (Symbol.iterator in Object(t) || "[object Arguments]" === Object.prototype.toString.call(t)) return Array.from(t) }(t) || function () { throw new TypeError("Invalid attempt to spread non-iterable instance") }() } function u(t, e) { for (var n = 0; n < e.length; n++) { var i = e[n]; i.enumerable = i.enumerable || !1, i.configurable = !0, "value" in i && (i.writable = !0), Object.defineProperty(t, i.key, i) } } var h = function () { function t(e, n) { !function (t, e) { if (!(t instanceof e)) throw new TypeError("Cannot call a class as a function") }(this, t), this.element = e, this.elementContainer = e, this.settings = n, this.isVisible = !0, this.isInit = !1, this.oldTranslateValue = -1, this.init = this.init.bind(this), a(e) ? this.init() : this.element.addEventListener("load", this.init) } var e, n, i; return e = t, (n = [{ key: "init", value: function () { var t = this; this.isInit || this.element.closest(".simpleParallax") || (!1 === this.settings.overflow && this.wrapElement(this.element), this.setTransformCSS(), this.getElementOffset(), this.intersectionObserver(), this.getTranslateValue(), this.animate(), this.settings.delay > 0 && setTimeout((function () { t.setTransitionCSS() }), 10), this.isInit = !0) } }, { key: "wrapElement", value: function () { var t = this.element.closest("picture") || this.element, e = document.createElement("div"); e.classList.add("simpleParallax"), e.style.overflow = "hidden", t.parentNode.insertBefore(e, t), e.appendChild(t), this.elementContainer = e } }, { key: "unWrapElement", value: function () { var t = this.elementContainer; t.replaceWith.apply(t, l(t.childNodes)) } }, { key: "setTransformCSS", value: function () { !1 === this.settings.overflow && (this.element.style[r] = "scale(".concat(this.settings.scale, ")")), this.element.style.willChange = "transform" } }, { key: "setTransitionCSS", value: function () { this.element.style.transition = "transform ".concat(this.settings.delay, "s ").concat(this.settings.transition) } }, { key: "unSetStyle", value: function () { this.element.style.willChange = "", this.element.style[r] = "", this.element.style.transition = "" } }, { key: "getElementOffset", value: function () { var t = this.elementContainer.getBoundingClientRect(); if (this.elementHeight = t.height, this.elementTop = t.top + s.positions.top, this.settings.customContainer) { var e = this.settings.customContainer.getBoundingClientRect(); this.elementTop = t.top - e.top + s.positions.top } this.elementBottom = this.elementHeight + this.elementTop } }, { key: "buildThresholdList", value: function () { for (var t = [], e = 1; e <= this.elementHeight; e++) { var n = e / this.elementHeight; t.push(n) } return t } }, { key: "intersectionObserver", value: function () { var t = { root: null, threshold: this.buildThresholdList() }; this.observer = new IntersectionObserver(this.intersectionObserverCallback.bind(this), t), this.observer.observe(this.element) } }, { key: "intersectionObserverCallback", value: function (t) { for (var e = t.length - 1; e >= 0; e--)t[e].isIntersecting ? this.isVisible = !0 : this.isVisible = !1 } }, { key: "checkIfVisible", value: function () { return this.elementBottom > s.positions.top && this.elementTop < s.positions.bottom } }, { key: "getRangeMax", value: function () { var t = this.element.clientHeight; this.rangeMax = t * this.settings.scale - t } }, { key: "getTranslateValue", value: function () { var t = ((s.positions.bottom - this.elementTop) / ((s.positions.height + this.elementHeight) / 100)).toFixed(1); return t = Math.min(100, Math.max(0, t)), 0 !== this.settings.maxTransition && t > this.settings.maxTransition && (t = this.settings.maxTransition), this.oldPercentage !== t && (this.rangeMax || this.getRangeMax(), this.translateValue = (t / 100 * this.rangeMax - this.rangeMax / 2).toFixed(0), this.oldTranslateValue !== this.translateValue && (this.oldPercentage = t, this.oldTranslateValue = this.translateValue, !0)) } }, { key: "animate", value: function () { var t, e = 0, n = 0; (this.settings.orientation.includes("left") || this.settings.orientation.includes("right")) && (n = "".concat(this.settings.orientation.includes("left") ? -1 * this.translateValue : this.translateValue, "px")), (this.settings.orientation.includes("up") || this.settings.orientation.includes("down")) && (e = "".concat(this.settings.orientation.includes("up") ? -1 * this.translateValue : this.translateValue, "px")), t = !1 === this.settings.overflow ? "translate3d(".concat(n, ", ").concat(e, ", 0) scale(").concat(this.settings.scale, ")") : "translate3d(".concat(n, ", ").concat(e, ", 0)"), this.element.style[r] = t } }]) && u(e.prototype, n), i && u(e, i), t }(); function c(t) { return function (t) { if (Array.isArray(t)) { for (var e = 0, n = new Array(t.length); e < t.length; e++)n[e] = t[e]; return n } }(t) || function (t) { if (Symbol.iterator in Object(t) || "[object Arguments]" === Object.prototype.toString.call(t)) return Array.from(t) }(t) || function () { throw new TypeError("Invalid attempt to spread non-iterable instance") }() } function f(t, e) { for (var n = 0; n < e.length; n++) { var i = e[n]; i.enumerable = i.enumerable || !1, i.configurable = !0, "value" in i && (i.writable = !0), Object.defineProperty(t, i.key, i) } } n.d(e, "default", (function () { return b })); var m, p, d, g = !0, v = !1, y = [], b = function () { function t(e, n) { !function (t, e) { if (!(t instanceof e)) throw new TypeError("Cannot call a class as a function") }(this, t), e && (this.elements = o(e), this.defaults = { delay: .4, orientation: "up", scale: 1.3, overflow: !1, transition: "cubic-bezier(0,0,0,1)", customContainer: !1, maxTransition: 0 }, this.settings = Object.assign(this.defaults, n), "IntersectionObserver" in window || (g = !1), this.settings.customContainer && (console.log(o(this.settings.customContainer)[0]), this.customContainer = o(this.settings.customContainer)[0]), this.lastPosition = -1, this.resizeIsDone = this.resizeIsDone.bind(this), this.handleResize = this.handleResize.bind(this), this.proceedRequestAnimationFrame = this.proceedRequestAnimationFrame.bind(this), this.init()) } var e, n, i; return e = t, (n = [{ key: "init", value: function () { var t = this; s.setViewportAll(this.customContainer), y = [].concat(c(this.elements.map((function (e) { return new h(e, t.settings) }))), c(y)), m = y.length, v || (this.proceedRequestAnimationFrame(), window.addEventListener("resize", this.resizeIsDone), v = !0) } }, { key: "resizeIsDone", value: function () { clearTimeout(d), d = setTimeout(this.handleResize, 500) } }, { key: "handleResize", value: function () { s.setViewportAll(this.customContainer); for (var t = m - 1; t >= 0; t--)y[t].getElementOffset(), y[t].getRangeMax(); this.lastPosition = -1 } }, { key: "proceedRequestAnimationFrame", value: function () { if (s.setViewportTop(this.customContainer), this.lastPosition !== s.positions.top) { s.setViewportBottom(); for (var t = m - 1; t >= 0; t--)this.proceedElement(y[t]); p = window.requestAnimationFrame(this.proceedRequestAnimationFrame), this.lastPosition = s.positions.top } else p = window.requestAnimationFrame(this.proceedRequestAnimationFrame) } }, { key: "proceedElement", value: function (t) { (!g || this.customContainer ? t.checkIfVisible() : t.isVisible) && t.getTranslateValue() && t.animate() } }, { key: "destroy", value: function () { var t = this, e = []; y = y.filter((function (n) { return t.elements.includes(n.element) ? (e.push(n), !1) : n })); for (var n = e.length - 1; n >= 0; n--)e[n].unSetStyle(), !1 === this.settings.overflow && e[n].unWrapElement(); (m = y.length) || (window.cancelAnimationFrame(p), window.removeEventListener("resize", this.handleResize)) } }]) && f(e.prototype, n), i && f(e, i), t }() }]).default }));


!function (n, t) { "object" == typeof exports && "undefined" != typeof module ? module.exports = t() : "function" == typeof define && define.amd ? define(t) : n.Splitting = t() }(this, function () {
  "use strict"
  var u = document, l = u.createTextNode.bind(u)
  function d(n, t, e) { n.style.setProperty(t, e) } function f(n, t) { return n.appendChild(t) } function p(n, t, e, r) {
    var i = u.createElement("span")
    return t && (i.className = t), e && (!r && i.setAttribute("data-" + t, e), i.textContent = e), n && f(n, i) || i
  } function h(n, t) { return n.getAttribute("data-" + t) } function m(n, t) { return n && 0 != n.length ? n.nodeName ? [n] : [].slice.call(n[0].nodeName ? n : (t || u).querySelectorAll(n)) : [] } function o(n) {
    for (var t = []; n--;)t[n] = []
    return t
  } function g(n, t) { n && n.some(t) } function c(t) { return function (n) { return t[n] } } var a = {}
  function n(n, t, e, r) { return { by: n, depends: t, key: e, split: r } } function e(n) {
    return function t(e, n, r) {
      var i = r.indexOf(e)
      if (-1 == i) r.unshift(e), g(a[e].depends, function (n) { t(n, e, r) })
      else {
        var u = r.indexOf(n)
        r.splice(i, 1), r.splice(u, 0, e)
      } return r
    }(n, 0, []).map(c(a))
  } function t(n) { a[n.by] = n } function v(n, r, i, u, o) {
    n.normalize()
    var c = [], a = document.createDocumentFragment()
    u && c.push(n.previousSibling)
    var s = []
    return m(n.childNodes).some(function (n) {
      if (!n.tagName || n.hasChildNodes()) {
        if (n.childNodes && n.childNodes.length) return s.push(n), void c.push.apply(c, v(n, r, i, u, o))
        var t = n.wholeText || "", e = t.trim()
        e.length && (" " === t[0] && s.push(l(" ")), g(e.split(i), function (n, t) {
          t && o && s.push(p(a, "whitespace", " ", o))
          var e = p(a, r, n)
          c.push(e), s.push(e)
        }), " " === t[t.length - 1] && s.push(l(" ")))
      } else s.push(n)
    }), g(s, function (n) { f(a, n) }), n.innerHTML = "", f(n, a), c
  } var s = 0
  var i = "words", r = n(i, s, "word", function (n) { return v(n, "word", /\s+/, 0, 1) }), y = "chars", w = n(y, [i], "char", function (n, e, t) {
    var r = []
    return g(t[i], function (n, t) { r.push.apply(r, v(n, "char", "", e.whitespace && t)) }), r
  })
  function b(t) {
    var f = (t = t || {}).key
    return m(t.target || "[data-splitting]").map(function (a) {
      var s = a["ðŸŒ"]
      if (!t.force && s) return s
      s = a["ðŸŒ"] = { el: a }
      var n = e(t.by || h(a, "splitting") || y), l = function (n, t) {
        for (var e in t) n[e] = t[e]
        return n
      }({}, t)
      return g(n, function (n) {
        if (n.split) {
          var t = n.by, e = (f ? "-" + f : "") + n.key, r = n.split(a, l, s)
          e && (i = a, c = (o = "--" + e) + "-index", g(u = r, function (n, t) { Array.isArray(n) ? g(n, function (n) { d(n, c, t) }) : d(n, c, t) }), d(i, o + "-total", u.length)), s[t] = r, a.classList.add(t)
        } var i, u, o, c
      }), a.classList.add("splitting"), s
    })
  } function N(n, t, e) {
    var r = m(t.matching || n.children, n), i = {}
    return g(r, function (n) { var t = Math.round(n[e]); (i[t] || (i[t] = [])).push(n) }), Object.keys(i).map(Number).sort(x).map(c(i))
  } function x(n, t) { return n - t } b.html = function (n) {
    var t = (n = n || {}).target = p()
    return t.innerHTML = n.content, b(n), t.outerHTML
  }, b.add = t
  var T = n("lines", [i], "line", function (n, t, e) { return N(n, { matching: e[i] }, "offsetTop") }), L = n("items", s, "item", function (n, t) { return m(t.matching || n.children, n) }), k = n("rows", s, "row", function (n, t) { return N(n, t, "offsetTop") }), A = n("cols", s, "col", function (n, t) { return N(n, t, "offsetLeft") }), C = n("grid", ["rows", "cols"]), M = "layout", S = n(M, s, s, function (n, t) {
    var e = t.rows = +(t.rows || h(n, "rows") || 1), r = t.columns = +(t.columns || h(n, "columns") || 1)
    if (t.image = t.image || h(n, "image") || n.currentSrc || n.src, t.image) {
      var i = m("img", n)[0]
      t.image = i && (i.currentSrc || i.src)
    } t.image && d(n, "background-image", "url(" + t.image + ")")
    for (var u = e * r, o = [], c = p(s, "cell-grid"); u--;) {
      var a = p(c, "cell")
      p(a, "cell-inner"), o.push(a)
    } return f(n, c), o
  }), H = n("cellRows", [M], "row", function (n, t, e) {
    var r = t.rows, i = o(r)
    return g(e[M], function (n, t, e) { i[Math.floor(t / (e.length / r))].push(n) }), i
  }), O = n("cellColumns", [M], "col", function (n, t, e) {
    var r = t.columns, i = o(r)
    return g(e[M], function (n, t) { i[t % r].push(n) }), i
  }), j = n("cells", ["cellRows", "cellColumns"], "cell", function (n, t, e) { return e[M] })
  return t(r), t(w), t(T), t(L), t(k), t(A), t(C), t(S), t(H), t(O), t(j), b
});



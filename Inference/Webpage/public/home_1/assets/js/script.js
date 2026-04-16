(function () {
  gsap.registerPlugin(ScrollTrigger, ScrollSmoother);

  const smoother = ScrollSmoother.create({
    content: "#scrollsmoother-container",
    smooth: 2,
    normalizeScroll: true,
    ignoreMobileResize: true,
    effects: true,
  });

  let headings = gsap.utils.toArray(".js-title").reverse();
  headings.forEach((heading, i) => {
    let headingIndex = i + 1;
    let mySplitText = new SplitText(heading, { type: "chars" });
    let chars = mySplitText.chars;

    chars.forEach((char, i) => {
      smoother.effects(char, { lag: (i + headingIndex) * 0.1, speed: 1 });
    });
  });


  let splitTextLines = gsap.utils.toArray(".js-splittext-lines");

  splitTextLines.forEach(splitTextLine => {
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: splitTextLine,
        start: 'top 90%',
        duration: 2,
        end: 'bottom 60%',
        scrub: false,
        markers: false,
        toggleActions: 'play none none none'
      }
    });

    const itemSplitted = new SplitText(splitTextLine, { type: "lines" });
    gsap.set(splitTextLine, { perspective: 400 });
    itemSplitted.split({ type: "lines" })
    tl.from(itemSplitted.lines, { duration: 1, delay: 0.5, opacity: 0, rotationX: -80, force3D: true, transformOrigin: "top center -50", stagger: 0.1 });
  });
})();
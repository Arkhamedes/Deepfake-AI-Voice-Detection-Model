function offset(element) {
    const box = element.getBoundingClientRect();
    return {
        top: box.top + (window.scrollY || document.documentElement.scrollTop) - (document.documentElement.clientTop || 0),
        left: box.left + (window.scrollX || document.documentElement.scrollLeft) - (document.documentElement.clientLeft || 0)
    }
}
// ------------ gsap scripts -----------
(function () {
    gsap.registerPlugin(ScrollTrigger, ScrollSmoother);

    // const smoother = ScrollSmoother.create({
    //     content: "#scrollsmoother-container",
    //     smooth: 2,
    //     normalizeScroll: true,
    //     ignoreMobileResize: true,
    //     effects: true
    // });

    // let headings = gsap.utils.toArray(".js-title").reverse();
    // headings.forEach((heading, i) => {
    //     let headingIndex = i + 1;
    //     let mySplitText = new SplitText(heading, { type: "chars" });
    //     let chars = mySplitText.chars;

    //     chars.forEach((char, i) => {
    //         smoother.effects(char, { lag: (i + headingIndex) * 0.1, speed: 1 });
    //     });
    // });


    let splitTextLines = gsap.utils.toArray(".js-splittext-lines");

    splitTextLines.forEach(splitTextLine => {
        const tl = gsap.timeline({
            scrollTrigger: {
                trigger: splitTextLine,
                start: 'top 90%',
                duration: 1,
                end: 'bottom 60%',
                scrub: false,
                markers: false,
                toggleActions: 'play none none none'
            }
        });

        const itemSplitted = new SplitText(splitTextLine, { type: "lines" });
        gsap.set(splitTextLine, { perspective: 400 });
        itemSplitted.split({ type: "lines" })
        tl.from(itemSplitted.lines, { duration: 1, delay: 0, opacity: 0, rotationX: -80, force3D: true, transformOrigin: "top center -50", stagger: 0.1 });
    });

})();

(function () {
    //  Button Move Animation
    const all_btns = gsap.utils.toArray(".btn_wrapper");
    if (all_btns.length > 0) {
        var all_btn = gsap.utils.toArray(".btn_wrapper");
    }
    else {
        var all_btn = gsap.utils.toArray("#btn_wrapper");
    }
    const all_btn_cirlce = gsap.utils.toArray(".btn-item");
    all_btn.forEach((btn, i) => {
        btn.addEventListener("mousemove", function (e) {
            parallaxIt(e, all_btn_cirlce[i], 50);
        });

        function parallaxIt(e, target, movement) {
            var $this = btn;
            var relX = e.pageX - offset($this).left;
            var relY = e.pageY - offset($this).top;

            gsap.to(target, 0.5, {
                x: ((relX - $this.getBoundingClientRect().width / 2) / $this.getBoundingClientRect().width) * movement,
                y: ((relY - $this.getBoundingClientRect().height / 2) / $this.getBoundingClientRect().height) * movement,
                ease: Power2.easeOut,
            });
        }
        btn.addEventListener("mouseleave", function (e) {
            gsap.to(all_btn_cirlce[i], 0.5, {
                x: 0,
                y: 0,
                ease: Power2.easeOut,
            });
        });
    });
})();


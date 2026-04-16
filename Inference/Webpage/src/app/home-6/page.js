import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
import ScrollToTop from "@/components/Common/ScrollToTop";
//= Page Components
import TopNavbar from "@/components/BusinessShowcase/TopNavbar";
import Header from "@/components/BusinessShowcase/Header";
import About from "@/components/BusinessShowcase/About";
import Services from "@/components/BusinessShowcase/Services";
import Projects from "@/components/BusinessShowcase/Projects";
import Marque from "@/components/BusinessShowcase/Marque";
import Clients from "@/components/BusinessShowcase/Clients";
import Revenue from "@/components/BusinessShowcase/Revenue";
import Testimonials from "@/components/BusinessShowcase/Testimonials";
import Footer from "@/components/BusinessShowcase/Footer";

export const metadata = {
  title: 'Swak - Business Showcase',
  icons: {
    icon: "/home_6/assets/img/fav2.png",
    shortcut: "/home_6/assets/img/fav2.png",
    other: generateStylesheetObject([
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/all.min.css',
      '/common/assets/css/lib/line-awesome.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/common_style.css',
      '/home_6/assets/css/home_6_style.css'
    ])
  }
}

export default function Home6() {
  return (
    <body className="home-style6">
      <LoadingScreen />
      {/* Page Components */}
      <div id="scrollsmoother-container">
        <TopNavbar />
        <Header />
        <main>
          <About />
          <Services />
          <Projects />
          <Marque />
          <Clients />
          <Revenue />
          <Testimonials />
        </main>
        <Footer />
      </div>
      <MouseCursor />
      <ScrollToTop />
      {/* Page Scripts */}
      <Script src="/common/assets/js/lib/bootstrap.bundle.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/gsap.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollSmoother.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollTrigger.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/SplitText.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/common_js.js" strategy="lazyOnload" />
      <Script src="/home_6/assets/js/home_6_scripts.js" strategy="lazyOnload" />
    </body>
  )
}

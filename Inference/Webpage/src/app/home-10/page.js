import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
import ScrollToTop from "@/components/Common/ScrollToTop";
//= Page Components
import SideMenu from "@/components/TechLanding/SideMenu";
import Header from "@/components/TechLanding/Header";
import Services from "@/components/TechLanding/Services";
import About from "@/components/TechLanding/About";
import Clients from "@/components/TechLanding/Clients";
import Portfolio from "@/components/TechLanding/Portfolio";
import FAQ from "@/components/TechLanding/FAQ";
import Testimonials from "@/components/TechLanding/Testimonials";
import Blog from "@/components/TechLanding/Blog";
import Footer from "@/components/TechLanding/Footer";

export const metadata = {
  title: 'Swak - Tech Landing',
  icons: {
    icon: "/home_10/assets/img/fav2.png",
    shortcut: "/home_10/assets/img/fav2.png",
    other: generateStylesheetObject([
      'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap',
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/all.min.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/line-awesome.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/common_style.css',
      '/home_10/assets/css/home_10_style.css'
    ])
  }
}

export default function Home10() {
  return (
    <body className="home-style10">
      <LoadingScreen />
      {/* Page Components */}
      <SideMenu />
      <div id="scrollsmoother-container">
        <Header />
        <main>
          <Services />
          <About />
          <Clients />
          <Portfolio />
          <FAQ />
          <Testimonials />
          <Blog />
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
      <Script src="/common/assets/js/lib/charming.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/lib/TweenMax.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/lib/demo.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/common_js.js" strategy="lazyOnload" />
      <Script src="/home_10/assets/js/home_10_scripts.js" strategy="lazyOnload" />
    </body>
  )
}

import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
import ScrollToTop from "@/components/Common/ScrollToTop";
//= Page Components
import Navbar from "@/components/ModernStartup/Navbar";
import Header from "@/components/ModernStartup/Header";
import About from "@/components/ModernStartup/About";
import Services from "@/components/ModernStartup/Services";
import Projects from "@/components/ModernStartup/Projects";
import Partners from "@/components/ModernStartup/Partners";
import Testimonials from "@/components/ModernStartup/Testimonials";
import Pricing from "@/components/ModernStartup/Pricing";
import FAQ from "@/components/ModernStartup/FAQ";
import Approach from "@/components/ModernStartup/Approach";
import Blog from "@/components/ModernStartup/Blog";
import Contact from "@/components/ModernStartup/Contact";
import Footer from "@/components/ModernStartup/Footer";

export const metadata = {
  title: 'Swak - Modern Startup',
  icons: {
    icon: "/home_3/assets/img/fav2.png",
    shortcut: "/home_3/assets/img/fav2.png",
    other: generateStylesheetObject([
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/all.min.css',
      '/common/assets/css/lib/line-awesome.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/common_style.css',
      '/home_3/assets/css/home_3_style.css'
    ])
  }
}

export default function Home3() {
  return (
    <body className="home-style3">
      <LoadingScreen />
      {/* Page Components */}
      <div id="scrollsmoother-container">
        <Navbar />
        <Header />
        <main>
          <About />
          <Services />
          <Projects />
          <Partners />
          <Testimonials />
          <Pricing />
          <FAQ />
          <Approach />
          <Blog />
          <Contact />
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
      <Script src="/home_3/assets/js/home_3_scripts.js" strategy="lazyOnload" />
    </body>
  )
}

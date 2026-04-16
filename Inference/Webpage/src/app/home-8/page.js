import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
//= Page Components
import Navbar from "@/components/PersonalPortfolio/Navbar";
import Header from "@/components/PersonalPortfolio/Header";
import Services from "@/components/PersonalPortfolio/Services";
import FeaturesSlider from "@/components/PersonalPortfolio/FeaturesSlider";
import Partners from "@/components/PersonalPortfolio/Partners";
import Portfolio from "@/components/PersonalPortfolio/Portfolio";
import Contact from "@/components/PersonalPortfolio/Contact";
import Testimonials from "@/components/PersonalPortfolio/Testimonials";
import Awards from "@/components/PersonalPortfolio/Awards";
import Blog from "@/components/PersonalPortfolio/Blog";
import Footer from "@/components/PersonalPortfolio/Footer";

export const metadata = {
  title: 'Swak - Personal Portfolio',
  icons: {
    icon: "/home_8/assets/img/fav2.png",
    shortcut: "/home_8/assets/img/fav2.png",
    other: generateStylesheetObject([
      'https://fonts.googleapis.com/css2?family=Amiri:ital,wght@0,400;0,700;1,400;1,700&display=swap',
      'https://fonts.googleapis.com/css2?family=Prata&display=swap',
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/all.min.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/common_style.css',
      '/home_8/assets/css/home_8_style.css'
    ])
  }
}

export default function Home8() {
  return (
    <body className="home-style8">
      <LoadingScreen />
      {/* Page Components */}
      <div id="scrollsmoother-container">
        <Navbar />
        <Header />
        <main>
          <Services />
          <FeaturesSlider />
          <Partners />
          <Portfolio />
          <Contact />
          <Testimonials />
          <Awards />
          <Blog />
        </main>
        <Footer />
      </div>
      <MouseCursor />
      {/* Page Scripts */}
      <Script src="/common/assets/js/lib/bootstrap.bundle.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/gsap.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollSmoother.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/ScrollTrigger.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/gsap_lib/SplitText.min.js" strategy="lazyOnload" />
      <Script src="/common/assets/js/common_js.js" strategy="lazyOnload" />
      <Script src="/home_8/assets/js/home_8_scripts.js" strategy="lazyOnload" />
    </body>
  )
}

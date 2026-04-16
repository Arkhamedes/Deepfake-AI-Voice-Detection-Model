import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import MouseCursor from "@/components/Common/MouseCursor";
//= Page Components
import TopNavbar from "@/components/ShopHome/TopNavbar";
import Navbar from "@/components/ShopHome/Navbar";
import Header from "@/components/ShopHome/Header";
import Products from "@/components/ShopHome/Products";
import ServicesSlider from "@/components/ShopHome/ServicesSlider";
import Newsletter from "@/components/ShopHome/Newsletter";
import Testimonials from "@/components/ShopHome/Testimonials";
import Partners from "@/components/ShopHome/Partners";
import Blog from "@/components/ShopHome/Blog";
import Footer from "@/components/ShopHome/Footer";

export const metadata = {
  title: 'Swak - Shop Home',
  icons: {
    icon: "/home_9/assets/img/fav2.png",
    shortcut: "/home_9/assets/img/fav2.png",
    other: generateStylesheetObject([
      'https://fonts.googleapis.com/css2?family=Rubik:wght@300;400;500;600;700;800;900&display=swap',
      '/common/assets/css/lib/bootstrap.min.css',
      '/common/assets/css/lib/all.min.css',
      '/common/assets/css/lib/themify-icons.css',
      '/common/assets/css/lib/ionicons.css',
      '/common/assets/css/lib/animate.css',
      '/common/assets/css/lib/swiper8.min.css',
      '/common/assets/css/common_style.css',
      '/home_9/assets/css/home_9_style.css'
    ])
  }
}

export default function Home9() {
  return (
    <body className="home-style9">
      <LoadingScreen />
      {/* Page Components */}
      <div id="scrollsmoother-container">
        <TopNavbar />
        <Navbar />
        <Header />
        <main>
          <Products />
          <ServicesSlider />
          <Newsletter />
          <Testimonials />
          <Partners />
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
      <Script src="/home_9/assets/js/home_9_scripts.js" strategy="lazyOnload" />
    </body>
  )
}

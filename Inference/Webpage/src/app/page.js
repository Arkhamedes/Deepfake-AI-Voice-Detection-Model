import Script from "next/script";
//= Scripts
import generateStylesheetObject from '@/common/generateStylesheetsObject';
//= Common Components
import LoadingScreen from "@/components/Common/Loader";
import Cursor from "@/components/Common/Cursor";
import ProgressScroll from "@/components/Common/ProgressScroll";
//= Page Components
import Navbar from "@/components/Preview/Navbar";
import Overlay from "@/components/Preview/Overlay";
import Header from "@/components/Preview/Header";
import Demos from "@/components/Preview/Demos";
import Marq from "@/components/Preview/Marq";
import Pages from "@/components/Preview/Pages";
import ShopPages from "@/components/Preview/ShopPages";
import CallToAction from "@/components/Preview/CallToAction";
import Footer from "@/components/Preview/Footer";
import ThemeSwitcher from "@/components/Common/ThemeSwitcher";

export const metadata = {
  title: 'Swak - Preview',
  icons: {
    icon: "/landing-preview/img/fav.png",
    shortcut: "/landing-preview/img/fav.png",
    other: generateStylesheetObject([
      '/landing-preview/css/plugins.css',
      '/landing-preview/css/style.css',
      '/landing-preview/css/preview-style.css',
    ])
  }
}

export default function Preview() {
  return (
    <body className="dark-theme">
      <LoadingScreen />
      <Cursor />
      <ProgressScroll />

      <Navbar />
      <Overlay />
      <main>
        <Header />
        <Demos />
        <Marq />
        <Pages />
        <ShopPages />
        <CallToAction />
        <Footer />
      </main>
      <ThemeSwitcher />

      <Script src="/landing-preview/js/bootstrap.bundle.min.js" strategy="lazyOnload" />
      <Script src="/landing-preview/js/wow.min.js" strategy="lazyOnload" />
      <Script src="/landing-preview/js/gsap.min.js" strategy="lazyOnload" />
      <Script src="/landing-preview/js/ScrollTrigger.min.js" strategy="lazyOnload" />
      <Script src="/landing-preview/js/plugins.js" strategy="lazyOnload" />
    </body>
  )
}

"use client";
import React, { useEffect } from 'react';
//= Scripts
import mouseEffect from '@/common/mouseEffect';

function MouseCursor() {
  useEffect(() => {
    mouseEffect();
  }, []);

  return (
    <>
      <div className="mouse-cursor cursor-outer"></div>
      <div className="mouse-cursor cursor-inner"></div>
    </>
  )
}

export default MouseCursor
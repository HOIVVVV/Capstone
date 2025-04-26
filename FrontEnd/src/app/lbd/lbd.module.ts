import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { LbdChartComponent } from './lbd-chart/lbd-chart.component'; // ✅ lbd-chart 컴포넌트 import

@NgModule({
  declarations: [
    LbdChartComponent
  ],
  imports: [
    CommonModule
  ],
  exports: [
    LbdChartComponent // ✅ 외부 모듈에서 사용할 수 있게 export
  ]
})
export class LbdModule { }
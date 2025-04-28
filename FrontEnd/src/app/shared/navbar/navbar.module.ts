import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';      // ✅ 추가
import { NavbarComponent } from './navbar.component';

@NgModule({
  imports: [
    CommonModule,
    RouterModule,
    FormsModule                                    // ✅ 반드시 포함
  ],
  declarations: [ NavbarComponent ],
  exports: [ NavbarComponent ]
})
export class NavbarModule {}
use crate::rect::Rect;

pub struct Ray {
    pub start: cgmath::Vector2<f32>,
    pub dir: cgmath::Vector2<f32>,
}

impl Ray {
    pub fn intersects(&self, rect: &Rect) -> bool {
        let mut t_min = -f32::INFINITY;
        let mut t_max = f32::INFINITY;

        if self.dir.x != 0.0 {
            let tx1 = (rect.min.x - self.start.x) / self.dir.x;
            let tx2 = (rect.max.x - self.start.x) / self.dir.x;

            t_min = t_min.max(tx1.min(tx2));
            t_max = t_max.min(tx1.max(tx2));
        }

        if self.dir.y != 0.0 {
            let ty1 = (rect.min.y - self.start.y) / self.dir.y;
            let ty2 = (rect.max.y - self.start.y) / self.dir.y;

            t_min = t_min.max(ty1.min(ty2));
            t_max = t_max.min(ty1.max(ty2));
        }

        let hit = t_max >= 0.0 && t_max >= t_min;

        if hit {
            println!("Hit @ {:?}", t_min * self.dir + self.start);
        }

        hit
    }
}

import type { Span } from "../api";

type Product = { code: string | null; name: string; price: string; description: string; image: string };
type Order = { lines: string[]; subtotal: string; order_codes: string };

// Liora's WhatsApp number from the catalogue (85913 53636), in international format.
const WHATSAPP = "918591353636";

function whatsappLink(order: string): string {
  return `https://wa.me/${WHATSAPP}?text=${encodeURIComponent(`Hi Liora, I'd like to order: ${order}`)}`;
}

function fromTools(spans: Span[]): { products: Product[]; order: Order | null } {
  const products: Product[] = [];
  let order: Order | null = null;
  for (const span of spans) {
    if (span.name !== "tools") continue;
    for (const { output } of span.metadata?.results ?? []) {
      if (output?.products) products.push(...output.products);
      if (output?.product) products.push(output.product);
      if (output?.lines) order = output;
    }
  }
  return { products, order };
}

/** Products and order totals returned by the catalogue tools, with a way to order each one. */
export function ProductCards({ spans }: { spans: Span[] }) {
  const { products, order } = fromTools(spans);
  if (!products.length && !order) return null;
  return (
    <div className="shop">
      {products.length > 0 && (
        <ul className="products">
          {products.map((p) => (
            <li key={p.name} className="product">
              <img src={p.image} alt="" loading="lazy" />
              <div className="product-body">
                {p.code && <span className="product-code">{p.code}</span>}
                <h4>{p.name}</h4>
                <p className="product-desc">{p.description}</p>
                <div className="product-foot">
                  <span className="product-price">{p.price}</span>
                  <a href={whatsappLink(`${p.code ?? p.name} x 1`)} target="_blank" rel="noreferrer">
                    Order on WhatsApp
                  </a>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
      {order && (
        <div className="order">
          <ul>
            {order.lines.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
          <p className="order-total">
            <span>{order.subtotal}</span> before shipping
          </p>
          <a className="primary order-link" href={whatsappLink(order.order_codes)} target="_blank" rel="noreferrer">
            Send this order on WhatsApp
          </a>
        </div>
      )}
    </div>
  );
}
